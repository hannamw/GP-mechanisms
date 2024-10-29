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