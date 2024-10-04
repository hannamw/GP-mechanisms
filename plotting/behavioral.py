#%%
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#%%
model = 'pythia-70m-deduped'
# maybe we want to change the display names of each model
model_name_mapping = {'pythia-70m-deduped': 'pythia-70m-deduped', 'gemma-2-2b': 'gemma-2-2b'}
df = pd.read_csv(f'../results/{model}/gp_with_probs.csv')
sns.set_theme(style='whitegrid')

means = {f'{ambiguity}_{token}': [df[df['condition'] == category][f'{ambiguity}_{token}_prob'].mean() for category in ['NPZ', 'NPS', 'MVRR']] for ambiguity in ['ambiguous','gp','post'] for token in ['punct', 'was']}

categories = ['NP/Z', 'NP/S', 'MV/RR']
x = np.arange(len(categories))  # the label locations
width = 0.13  # the width of the bars
multiplier = 0

fig, ax =  plt.subplots()
colors = ['lightcoral', 'firebrick', 'palegoldenrod', 'goldenrod', 'lightskyblue', 'dodgerblue']
column_labels = [f'{ambiguity}, p({token})' for ambiguity in ['Ambiguous', 'GP', 'Non-GP'] for token in ['GP', 'non-GP']]
for i, (attribute, measurement) in enumerate(means.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=column_labels[i], color=colors[i])
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probability')
ax.set_xlabel('Garden Path Structure')
ax.set_title(f'Garden Path Continuation Probabilities ({model_name_mapping[model]})')
ax.set_xticks(x + width * (len(means) - 1)/2, categories)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=3)
Path(f'behavioral').mkdir(exist_ok=True, parents=True)
plt.savefig(f'behavioral/{model}-behavioral.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')
plt.show()
# %%
