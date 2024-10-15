#%%
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

model = 'pythia-70m-deduped'
# maybe we want to change the display names of each model
model_name_mapping = {'pythia-70m-deduped': 'Pythia-70m', 'gemma-2-2b': 'Gemma-2-2B'}
df = pd.read_csv(f'../results/{model}/gp_with_probs.csv')
sns.set_theme(style='whitegrid')

means = {f'{ambiguity}_{token}': [df[df['condition'] == category][f'{ambiguity}_{token}_prob'].mean() for category in ['NPZ', 'NPS', 'MVRR']] for ambiguity in ['ambiguous','gp','post'] for token in ['punct', 'was']}
#means = {f'{ambiguity}_{token}': [df[df['condition'] == category][f'{ambiguity}_{token}_prob'].mean() for category in ['NPZ', 'NPS', 'MVRR']] for token in ['punct', 'was'] for ambiguity in ['ambiguous','gp','post']}

categories = ['NP/Z', 'NP/S', 'MV/RR']
x = np.arange(len(categories))  # the label locations
width = 0.13  # the width of the bars
multiplier = 0

fig, ax =  plt.subplots(figsize=(7.75,3))
colors = ['lightcoral', 'firebrick', 'palegoldenrod', 'goldenrod', 'lightskyblue', 'dodgerblue']
#colors = ['lightcoral', 'palegoldenrod', 'lightskyblue', 'firebrick', 'goldenrod', 'dodgerblue']
column_labels = [f'{ambiguity}, p({token})'  for token in ['GP', 'non-GP'] for ambiguity in ['Ambiguous', 'GP', 'Non-GP']]
handles = []
for i, (attribute, measurement) in enumerate(means.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=column_labels[i], color=colors[i])
    handles.append(rects)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probability')
ax.set_xlabel('Garden Path Structure')
ax.set_title(f'Garden Path Continuation Probabilities ({model_name_mapping[model]})')
ax.set_xticks(x + width * (len(means) - 1)/2, categories)

#handles = [handles[i] for i in [0,3,1,4,2,5]]
labels = [handle.get_label() for handle in handles]

leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
Path(f'behavioral').mkdir(exist_ok=True, parents=True)
plt.savefig(f'behavioral/{model}-behavioral.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')
plt.show()
# %%
