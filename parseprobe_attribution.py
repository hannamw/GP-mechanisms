#%%
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoTokenizer
from nnsight import LanguageModel

import attribution
import dictionary_learning as dl
from parseprobe import ParseProbe
#%%
DEVICE='cuda'
model_name = 'EleutherAI/pythia-70m-deduped'
model_name_noslash = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LanguageModel(model_name, device_map='auto')
model.eval()

attributions = defaultdict(dict)
for condition in ['NPZ', 'NPS']:
    df = pd.read_csv("data_csv/gp_same_len.csv")
    df = df[df['condition'] == condition]
    tokens = tokenizer(df['sentence_ambiguous'].tolist(), return_tensors='pt').to('cuda').input_ids
    tokens = torch.cat((torch.full((tokens.size(0),1), tokenizer.bos_token_id, device='cuda'), tokens) , dim=-1)
    
    # load probes
    probes = {'embed': ParseProbe()}
    for layer in range(model.config.num_hidden_layers):
        probes[layer] = ParseProbe()
        probes[layer].load_state_dict(torch.load(f"standalone_probes/layer{layer}.pt"))
    probes['embed'].load_state_dict(torch.load(f"standalone_probes/embeddings.pt"))

    for probe in probes.values():
        probe.to('cuda')

    if condition == 'NPZ':
        last_pos, verb_pos = 6, 4
    else:
        last_pos, verb_pos = 5, 3
        
    def get_probe_input(acts):
        # The probe takes in the activations of the 6th and 4th token, and decides what kind of arc to draw if any
        return torch.cat((acts[:, last_pos], acts[:, verb_pos]), dim=-1)
    
    for layer in ['embed', *range(6)]:
        if layer == 'embed':
            submodule = model.gpt_neox.embed_in
            state_dict = torch.load(f'feature-circuits-gp/dictionaries/pythia-70m-deduped/embed/10_32768/ae.pt')
        else:
            submodule = model.gpt_neox.layers[layer]
            state_dict = torch.load(f'feature-circuits-gp/dictionaries/pythia-70m-deduped/resid_out_layer{layer}/10_32768/ae.pt')
            
        submodules = [submodule]
        dictionaries = {submodule: dl.dictionary.AutoEncoder(512, 32768).to("cuda")}
        dictionaries[submodule].load_state_dict(state_dict)

        def metric_fn(model, labels=None):
            acts = submodule.output
            if layer != 'embed':
                acts = acts[0]
            probe_input = get_probe_input(acts)

            action_logits = probes[layer](probe_input)
            # 0 and 2 are the indices of the actions (GEN vs. right arc) that we want to contrast
            logit_diff = action_logits[:, 0] - action_logits[:, 2]

            return -logit_diff.mean()

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
        layer_name = 'embed' if layer == 'embed' else f'resid_{layer}'
        attributions[condition][layer_name] = effects[submodule]

torch.save(attributions, f'results/{model_name_noslash}/parse_probe/attribution.pt')
# %%
NPZ_circuit = torch.load('feature-circuits-gp/circuits/NPZ_ambiguous_samelen_dict10_node0.05_edge0.01_n24_aggnone.pt')
NPS_circuit = torch.load('feature-circuits-gp/circuits/NPS_ambiguous_samelen_dict10_node0.05_edge0.01_n24_aggnone.pt')

# %%
overlaps = {'NPZ': [], 'NPS': []}
for condition in ['NPZ', 'NPS']:
    nodes = NPZ_circuit['nodes'] if condition == 'NPZ' else NPS_circuit['nodes']
    print(condition)
    for layer, acts in attributions[condition].items():
        print(layer)
        topn = (nodes[layer].act.abs() > 0.1).sum().item()
        if topn > 0:
            circuit_locs, circuit_features = torch.unravel_index(nodes[layer].act.abs().view(-1).argsort()[-topn:], nodes[layer].act.shape)
            acts_mean = acts.act.mean(0)
            probe_locs, probe_features = torch.unravel_index(acts_mean.abs().view(-1).argsort()[-topn:], acts_mean.shape)
            probe_locs -= 1
            
            circuit_set = set(zip(circuit_locs.tolist(), circuit_features.tolist()))
            probe_set = set(zip(probe_locs.tolist(), probe_features.tolist()))
            intersection_size = len(circuit_set.intersection(probe_set))
            print(intersection_size, topn, intersection_size / topn)
            overlaps[condition].append(len(circuit_set.intersection(probe_set)) / topn)
# %%
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')
fig, ax  = plt.subplots(figsize=(5, 2))

for condition, overlap in overlaps.items():
    color = 'blue' if condition == 'NPZ' else 'orange'
    ax.plot(overlap, label=f'{condition}', color=color)
        
ax.set_xlabel('Layer')
ax.set_ylabel('Overlap (Recall with Circuit)')
ax.set_xticks(list(range(7)), ['embeds', *(str(x) for x in range(6))])
ax.set_title('Overlap Between Probe and Circuit Features')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
#zipped = list(zip(handles, labels))
#handles, labels = zip(*[zipped[i] for i in [0,3,1,4,2,5]])
leg = ax.legend(handles, labels, ncol=2, fancybox=True)

fig.show()
fig.savefig(f'results/{model_name_noslash}/parse_probe/probe_features.png', bbox_extra_artists=(leg,),bbox_inches='tight')
fig.savefig(f'results/{model_name_noslash}/parse_probe/probe_features.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')
# %%
