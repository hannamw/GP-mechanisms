#%%
from pathlib import Path 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch 

sns.set_style("whitegrid")

model_name = 'pythia-70m-deduped'
d = torch.load(f'../results/{model_name}/causal_probabilities.pt')

model_name_mapping = {'pythia-70m-deduped': 'pythia-70m-deduped', 'gemma-2-2b': 'gemma-2-2b'}
means = [[d[cond][struct][i].mean().item() for struct in ['NPZ', 'NPS']] for cond in ['intervened', 'baseline', 'random'] for i in [0, 1]]

categories = ['NP/Z', 'NP/S']
x = np.arange(len(categories))  # the label locations
width = 0.13  # the width of the bars
multiplier = 0


fig, ax =  plt.subplots(figsize=(7,3))
colors = ['lightcoral', 'firebrick', 'palegoldenrod', 'goldenrod', 'lightskyblue', 'dodgerblue']
column_labels = [f'{ambiguity}, p({token})' for ambiguity in ['Intervention', 'Baseline', 'Random Int.'] for token in ['GP', 'non-GP']]
for i, mean in enumerate(means):
    offset = width * multiplier
    rects = ax.bar(x + offset, mean, width, label=column_labels[i], color=colors[i])
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probability')
ax.set_xlabel('Garden Path Structure')
ax.set_title(f'Garden Path Continuation Probabilities ({model_name_mapping[model_name]})')
ax.set_xticks(x + width * (len(means) - 1)/2, categories)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)

Path(f'causal').mkdir(exist_ok=True, parents=True)
fig.savefig(f'causal/{model_name}-causal.png', bbox_extra_artists=(leg,),bbox_inches='tight')
fig.savefig(f'causal/{model_name}-causal.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')
plt.show()
# %%
