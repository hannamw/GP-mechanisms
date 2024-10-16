#%%
from pathlib import Path 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch 

sns.set_style("whitegrid")

model_name = 'gemma-2-2b'
# model_name = 'gemma-2-2b'
d = torch.load(f'../results/{model_name}/causal_probabilities.pt')

model_name_mapping = {'pythia-70m-deduped': 'Pythia-70m', 'gemma-2-2b': 'Gemma-2-2B'}
means = [[d[cond][struct][i].mean().item() for cond in ['intervened', 'baseline', 'random'] for i in [0, 1]] for struct in ['NPZ', 'NPS']]

categories = ['NP/Z', 'NP/S']
x = np.arange(len(categories))  # the label locations
width = 0.1  # the width of the bars
extra_offset = 0.06


fig, axs =  plt.subplots(ncols = 2, sharey=True, figsize=(7.5,2.3))
#colors = ['lightcoral', 'firebrick', 'palegoldenrod', 'goldenrod', 'lightskyblue', 'dodgerblue']

colors = ['lightcoral', 'firebrick', 'gray', 'black', 'lightskyblue', 'dodgerblue']
column_labels = [f'{ambiguity}, p({token})' for ambiguity in ['Intervention', 'No Int.', 'Random Int.'] for token in ['GP', 'non-GP']]

handles = []
for struct_mean, structure, ax in zip(means, ['NP/Z', 'NP/S'], axs):
    multiplier = 0
    offsets = [(i-1)* width + (i//2) * extra_offset for i in range(6)]
    #offsets = [(i-1)* width for i in range(6)]
    for i, measurement in enumerate(struct_mean):
        print(structure, column_labels[i], measurement)
        rects = ax.bar(offsets[i], measurement, width, label=column_labels[i], color=colors[i], edgecolor='black')
        handles.append(rects)
        multiplier += 1
        
    ax.set_title(f'{structure}', y=-0.15)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Add some text for labels, title and custom x-axis tick labels, etc.
axs[0].set_ylabel('Probability')
#ax.set_xlabel('Garden Path Structure')
suptitle = fig.suptitle(f'Garden Path Continuation Probabilities ({model_name_mapping[model_name]})')
#ax.set_xticks(x + width * (len(means) - 1)/2, categories)
handles = handles[:6]
labels = [handle.get_label() for handle in handles]
leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3,
                 frameon=False)

# stylistic details
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.grid(False)

Path(f'causal-subplots').mkdir(exist_ok=True, parents=True)
fig.savefig(f'causal-subplots/{model_name}-causal-subplots.png', bbox_extra_artists=(leg,suptitle),bbox_inches='tight')
fig.savefig(f'causal-subplots/{model_name}-causal-subplots.pdf', bbox_extra_artists=(leg,suptitle), bbox_inches='tight')
plt.show()
# %%
