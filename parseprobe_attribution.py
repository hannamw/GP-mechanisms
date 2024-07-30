#%%
import importlib  
import sys
sys.path.append('feature-circuits-gp')

import torch
from torch import optim
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel

import attribution
import dictionary_learning as dl
from parseprobe import ParseProbe
#%%
DEVICE='cuda'
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

tokens = torch.cat((torch.full((tokens.size(0),1), tokenizer.bos_token_id, device='cuda'), tokens) , dim=-1)
# %%
# load probes
probes = {'embeds': ParseProbe()}
for layer in range(model.config.num_hidden_layers):
    probes[layer] = ParseProbe()
    probes[layer].load_state_dict(torch.load(f"standalone_probes/layer{layer}.pt"))
probes['embeds'].load_state_dict(torch.load(f"standalone_probes/embeddings.pt"))

for probe in probes.values():
    probe.to('cuda')

def get_probe_input(acts):
    # The probe takes in the activations of the 6th and 4th token, and decides what kind of arc to draw if any
    return torch.cat((acts[:, 6], acts[:, 4]), dim=-1)
#%%
layer = 4
submodules = [model.gpt_neox.layers[layer]]
dictionaries = {model.gpt_neox.layers[layer]: dl.dictionary.AutoEncoder(512, 32768).to("cuda")}
state_dict = torch.load(f'feature-circuits-gp/dictionaries/pythia-70m-deduped/resid_out_layer{layer}/10_32768/ae.pt')
dictionaries[model.gpt_neox.layers[layer]].load_state_dict(state_dict)

def metric_fn(model, labels=None):
    acts = model.gpt_neox.layers[layer].output[0]
    probe_input = get_probe_input(acts)

    action_logits = probes[layer](probe_input)
    # 0 and 2 are the indices of the actions (GEN vs. right arc) that we want to contrast
    logit_diff = action_logits[:, 0] - action_logits[:, 2]

    return logit_diff.mean()

# %%
effects, _, _, _ = attribution.patching_effect(
    tokens,
    None,
    model,
    submodules,
    dictionaries,
    metric_fn,
    metric_kwargs=dict(),
    method='attrib'
)
# %%
mean_effects = list(effects.values())[0].act.mean(0)
verb_effects = mean_effects[4]
noun_effects = mean_effects[6]
# %%
print(verb_effects.abs().argsort(descending=True)[:10])
print(noun_effects.abs().argsort(descending=True)[:10])
# %%
