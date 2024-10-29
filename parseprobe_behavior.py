#%%
from collections import defaultdict
import torch
import pandas as pd
from nnsight import LanguageModel
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from parseprobe import ParseProbe
#%%
DEVICE='cuda'
model_name = 'EleutherAI/pythia-70m-deduped'
model_name_noslash = model_name.split('/')[-1]
model = LanguageModel(model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

probes = {'embeds': ParseProbe()}
for layer in range(model.config.num_hidden_layers):
    probes[layer] = ParseProbe()
    probes[layer].load_state_dict(torch.load(f"standalone_probes/layer{layer}.pt"))
probes['embeds'].load_state_dict(torch.load(f"standalone_probes/embeddings.pt"))
for probe in probes.values():
    probe.to('cuda')

# change this to use either our dataset or Tiwa's (filtered for length)
orig_df = pd.read_csv("data_csv/gp_same_len.csv") 
#%%
probs = defaultdict(dict)
for condition in ['NPZ', 'NPS']:
    df = orig_df[orig_df['condition'] == condition]
    
    tokens = tokenizer(df['sentence_ambiguous'].tolist(), return_tensors='pt').to('cuda').input_ids

    tokens = torch.cat((torch.full((tokens.size(0),1), tokenizer.bos_token_id, device='cuda'), tokens) , dim=-1)

    if condition == 'NPZ':
        last_pos, verb_pos = 6, 4
    else:
        last_pos, verb_pos = 5, 3
    def get_probe_input(acts):
        # The probe takes in the activations of the 6th and 4th token, and decides what kind of arc to draw if any
        return torch.cat((acts[:, last_pos], acts[:, verb_pos]), dim=-1)
        #return torch.cat((acts[:, verb_pos], acts[:, last_pos]), dim=-1)
    
    condition_probs = []
    for layer in ['embeds', *range(6)]:
        if layer == 'embeds':
            long_layer = 'embed'
            submodule = model.gpt_neox.embed_in
        else:
            long_layer = f'resid_{layer}'
            submodule = model.gpt_neox.layers[layer]
        
        with model.trace(tokens):
            acts = submodule.output
            if layer != 'embeds':
                acts = acts[0]
            acts = acts.save()
        probe_input = get_probe_input(acts)
        probe_logits = probes[layer](probe_input).detach()
        probe_probs = torch.softmax(probe_logits, dim=-1).cpu()
        probs[condition][long_layer] = probe_probs

torch.save(probs, f'results/{model_name_noslash}/parse_probe/probe_probs.pt')
# %%
# NOTE: In reality, dimensions 0, 1, and 2 of the probe probs correspond to GEN, LEFT-ARC, and RIGHT-ARC, respectively.
# But, this is a little confusing: the proper order to feed the representations into the probe is [last, verb], but this is the opposite of how they appear in the sentence! This has to do with the overall parsing algorithm in the context of which the probe is used, I think
# So, LEFT-ARC means drawing an arc from the verb to the last token, and RIGHT-ARC means drawing an arc from the last token to the verb, the opposite of what you'd expect.

fig, ax  = plt.subplots(figsize=(5, 3))

for condition, data in probs.items():
    color = 'blue' if condition == 'NPZ' else 'orange'
    condition_data = torch.stack([data[layer] for layer in ['embed', *(f'resid_{i}' for i in range(6))]], dim=0)
    mean_data = condition_data.mean(1).numpy()
    for i, (action, linetype) in enumerate(zip(['GEN', 'RIGHT-ARC', 'LEFT-ARC'], ['-', '--', ':'])):
        ax.plot(mean_data[:, i], label=f'{condition} {action}', linestyle=linetype, color=color)
        
ax.set_xlabel('Layer')
ax.set_ylabel('Probability')
ax.set_xticks(list(range(7)), ['embeds', *(str(x) for x in range(6))])
ax.set_title('Probe Action Probabilities')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
zipped = list(zip(handles, labels))
handles, labels = zip(*[zipped[i] for i in [0,3,1,4,2,5]])
leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fancybox=True)

fig.show()
fig.savefig(f'results/{model_name_noslash}/parse_probe/probe_probs.png', bbox_extra_artists=(leg,),bbox_inches='tight')
fig.savefig(f'results/{model_name_noslash}/parse_probe/probe_probs.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')
# %%
