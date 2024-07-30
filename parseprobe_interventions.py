#%%
import torch
from torch import optim
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel

from parseprobe import ParseProbe
#%%
model_name = 'EleutherAI/pythia-70m'
model_name_noslash = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LanguageModel(model_name, device_map='auto')
model.eval()

NP_token = ','
Z_tokens = [' was', ' were']
NP_token_id = tokenizer(NP_token, add_special_tokens=False)['input_ids'][0]
NP_token_ids = [NP_token_id]
Z_token_ids = [tokenizer(z, add_special_tokens=False)['input_ids'][0] for z in Z_tokens]

# change this to use either our dataset or Tiwa's (filtered for length)
use_our_NPZ = True
if use_our_NPZ:
    df = pd.read_csv("data_csv/gp_same_len.csv")
    df = df[df['condition'] == 'NPZ']
    Z_label = [Z_token_ids[0] for _ in range(df.shape[0])]
    NP_label = [NP_token_id for _ in range(df.shape[0])]
    tokens = tokenizer(df['sentence_ambiguous'].tolist(), return_tensors='pt').to('cuda').input_ids
else:
    # with tiwa's dataset, sometimes the subject of the following clause is plural, and we need to measure the logit of ' were' instead of ' was'
    df = pd.read_csv("data_csv/npz_tiwa_same_len.csv")
    Z_label = [Z_token_ids[1] if is_plural else Z_token_ids[0] for is_plural in df['plurality']]
    NP_label = [NP_token_id for _ in range(df.shape[0])]
    tokens = tokenizer(df['prefix'].tolist(), return_tensors='pt').to('cuda').input_ids
#%%
tokens = torch.cat((torch.full((tokens.size(0),1), tokenizer.bos_token_id, device='cuda'), tokens) , dim=-1)
probs = torch.softmax(model.trace(tokens, trace=False).logits.squeeze(), dim=-1)

# Looking at the default predictions w/o intervention
NP_probs = probs[:, -1, NP_label]
Z_probs = probs[:, -1, Z_label]
print(NP_probs.mean().cpu().item(), Z_probs.mean().cpu().item())
# %%
# storing all of the activations
activations = {}
with model.trace(tokens):
    activations['embeds'] = model.gpt_neox.embed_in.output.save()
    for layer in range(model.config.num_hidden_layers):
        activations[layer] = model.gpt_neox.layers[layer].output[0].detach().save()
# %%
# Loading up all the probes
probes = {'embeds': ParseProbe()}
for layer in range(model.config.num_hidden_layers):
    probes[layer] = ParseProbe()
    probes[layer].load_state_dict(torch.load(f"standalone_probes/layer{layer}.pt"))
probes['embeds'].load_state_dict(torch.load(f"standalone_probes/embeddings.pt"))

for probe in probes.values():
    probe.to('cuda')

# %%
def get_probe_input(acts):
    # The probe takes in the activations of the 6th and 4th token, and decides what kind of arc to draw if any
    return torch.cat((acts[:, 6], acts[:, 4]), dim=-1)

# Let's take a look at what the actions are by default
# 0: GEN, 1: LEFT-ARC, 2: RIGHT-ARC
for layer in ['embeds', *range(model.config.num_hidden_layers)]:
    probe_input = get_probe_input(activations[layer])
    actions = torch.softmax(probes[layer](probe_input), dim=-1)
    print(actions.mean(0).cpu().tolist())
# %%
target_action = 0
lr=0.001
patience=100
num_steps=50000
loss_tolerance=0.01
scheduler_patience=1000

# copy the activations to be updated
new_activations = {k: v.clone().detach() for k, v in activations.items()}
for v in new_activations.values():
    v.requires_grad = True
best_activations = {}
nll_loss = torch.nn.NLLLoss()

for layer in ['embeds', *range(model.config.num_hidden_layers)]:

    # this optimization loop is directly from Tiwa's code
    optimizer = torch.optim.Adam([new_activations[layer]], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=scheduler_patience
    )

    prediction_loss = 100  # Initialize the prediction loss as high
    increment_idx = 0

    smallest_loss = prediction_loss
    steps_since_best = 0

    while prediction_loss > loss_tolerance:
        probe_input = get_probe_input(new_activations[layer])
        log_action_distribution = torch.log_softmax(probes[layer](probe_input), dim=-1)

        # we're pushing the action towards 0 (GEN) rather than 2 (right-arc)
        target_action_tensor = torch.full((new_activations[layer].size(0),), target_action, dtype=torch.long, device='cuda')

        loss = nll_loss(log_action_distribution, target_action_tensor)

        prediction_loss = loss.clone().detach()
        if increment_idx == 0:
            initial_loss = loss.clone().detach()

        # optimize representations 
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if (smallest_loss - prediction_loss) > 0.001:
            best_activations[layer] = new_activations[layer].detach().clone()
            steps_since_best = 0
            smallest_loss = prediction_loss

        else:
            steps_since_best += 1
            # if steps_since_best == patience/2:
            if steps_since_best == patience:
                print("Breaking because of patience with loss", smallest_loss)
                break
        increment_idx += 1
    print(f"Exited grad update loop after {increment_idx} steps, ")
# %%
# observe how probe decisions change
for layer in ['embeds', *range(model.config.num_hidden_layers)]:
    new_probe_input = get_probe_input(new_activations[layer])
    new_action_distribution = torch.softmax(probes[layer](new_probe_input), dim=-1)
    print(new_action_distribution.mean(0))
# %%
# observe how the optimization changed model behavior
print("Orig:", NP_probs.mean().cpu().item(), Z_probs.mean().cpu().item())
# skipping embeds because they're annoying
for layer in range(model.config.num_hidden_layers):
    with model.trace(tokens):
        z = model.gpt_neox.layers[layer].output.save()
        orig_acts, x, y = model.gpt_neox.layers[layer].output
        orig_acts[:, 4] = new_activations[layer][:, 4]
        orig_acts[:, 6] = new_activations[layer][:, 6]
        model.gpt_neox.layers[layer].output = (orig_acts, x, y)
        new_logits = model.embed_out.output.save()
    new_probs = torch.softmax(new_logits.squeeze(), dim=-1)
    new_NP_probs = new_probs[:, -1, NP_token_ids].sum(-1)
    new_Z_probs = new_probs[:, -1, Z_token_ids].sum(-1)
    print(layer, new_NP_probs.mean().cpu().item(), new_Z_probs.mean().cpu().item())
# %%
