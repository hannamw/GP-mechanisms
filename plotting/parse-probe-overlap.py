#%%
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')

model_name = 'EleutherAI/pythia-70m-deduped'
model_name_noslash = model_name.split('/')[-1]
attributions = torch.load(f'../results/{model_name_noslash}/parse_probe/attribution.pt')
NPZ_circuit = torch.load('../feature-circuits-gp/circuits/NPZ_ambiguous_samelen_dict10_node0.05_edge0.01_n24_aggnone.pt')
NPS_circuit = torch.load('../feature-circuits-gp/circuits/NPS_ambiguous_samelen_dict10_node0.05_edge0.01_n24_aggnone.pt')

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
p = Path('parse-probe-overlap')
p.mkdir(exist_ok=True)
fig.savefig(f'parse-probe-overlap/{model_name_noslash}-parse-probe-overlap.png', bbox_extra_artists=(leg,),bbox_inches='tight')
fig.savefig(f'parse-probe-overlap/{model_name_noslash}-parse-probe-overlap.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')
# %%
