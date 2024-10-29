#%%
from pathlib import Path 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch 

from curlyBrace import curlyBrace

sns.set_style("whitegrid")

model_name = 'pythia-70m-deduped'
# model_name = 'gemma-2-2b'
d = torch.load(f'../results/{model_name}/causal_probabilities_largescale.pt')

model_name_mapping = {'pythia-70m-deduped': 'Pythia-70m', 'gemma-2-2b': 'Gemma-2-2B'}
# means = [[d[cond][struct][i].mean().item() for cond in ['intervened', 'baseline', 'random'] for i in [0, 1]] for struct in ['NPZ', 'NPS']]
means = [[],[]] 
all_means = []
for i, struct in enumerate(['NPZ', 'NPS']):
    for cond in ['intervened', 'random', 'baseline']:
        gp, nongp = list(zip(*d[cond][struct]))
        gp = torch.tensor(gp).mean().item()
        nongp = torch.tensor(nongp).mean().item()
        means[i].append(gp-nongp)
        all_means.append(gp - nongp)

categories = ['NP/Z', 'NP/S']
x = np.arange(len(categories))  # the label locations
width = 0.1  # the width of the bars
extra_offset = 0.06


fig, axs =  plt.subplots(ncols = 2, sharey=True, figsize=(7.5,2.3))
#colors = ['lightcoral', 'firebrick', 'palegoldenrod', 'goldenrod', 'lightskyblue', 'dodgerblue']

colors = ['lightcoral', 'lightskyblue', 'gray']
column_labels = [f'{ambiguity}' for ambiguity in ['Syntactic/Structural Features', 'Random Features', 'None']]

handles = []#axs[0].plot([], [], ' ', label="Sentence Structure:")[0]]
for struct_mean, structure, ax in zip(means, ['NP/Z', 'NP/S'], axs):
    multiplier = 0
    offsets = [(i-1)* width + (i) * extra_offset for i in range(3)]
    #offsets = [(i-1)* width for i in range(6)]
    for i, measurement in enumerate(struct_mean):
        print(structure, column_labels[i], measurement)
        rects = ax.bar(offsets[i], measurement, width, label=column_labels[i], color=colors[i], edgecolor='black')
        handles.append(rects)
        multiplier += 1
        
    ax.set_title(f'{structure}', y=-0.15)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Add some text for labels, title and custom x-axis tick labels, etc.
axs[0].set_ylabel('p(GP)-p(non-GP)')
#ax.set_xlabel('Garden Path Structure')
suptitle = fig.suptitle(f'Garden Path Continuation Probabilities ({model_name_mapping[model_name]}, Large-Scale)')
#ax.set_xticks(x + width * (len(means) - 1)/2, categories)
handles = handles[:3]
labels = [handle.get_label() for handle in handles]
leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, title='Intervention Type')

# stylistic details
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.grid(False)
    ax.axhline(0, color='black', linewidth=1.0)
    
extra_artists1 = curlyBrace(fig, axs[-1], (0.3, max(all_means)), (0.3,0.01), str_text='GP', color='black', clip_on=False, fontdict={'size':14})[-1]
extra_artists2 = curlyBrace(fig, axs[-1], (0.3, -0.01), (0.3,min(all_means)), str_text='non-GP', color='black', clip_on=False, fontdict={'size':14})[-1]
axs[-1].set_xlim(axs[-2].get_xlim())

Path(f'causal-subplots-largescale-difference').mkdir(exist_ok=True, parents=True)
fig.savefig(f'causal-subplots-largescale-difference/{model_name}-causal-subplots-largescale-difference.png', bbox_extra_artists=(leg,suptitle, *extra_artists1, *extra_artists2),bbox_inches='tight')
fig.savefig(f'causal-subplots-largescale-difference/{model_name}-causal-subplots-largescale-difference.pdf', bbox_extra_artists=(leg,suptitle, *extra_artists1, *extra_artists2), bbox_inches='tight')
plt.show()
# %%
