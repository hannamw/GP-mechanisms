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

means = [{f'{ambiguity}_{token}': [df[df['condition'] == category][f'{ambiguity}_{token}_prob'].mean() ]for ambiguity in ['ambiguous','gp','post']  for token in ['punct', 'was'] } for category in ['NPZ', 'NPS', 'MVRR']]
#means = {f'{ambiguity}_{token}': [df[df['condition'] == category][f'{ambiguity}_{token}_prob'].mean() for category in ['NPZ', 'NPS', 'MVRR']] for token in ['punct', 'was'] for ambiguity in ['ambiguous','gp','post']}

categories = ['NP/Z', 'NP/S', 'MV/RR']
x = np.array([0])  # the label locations
width = 0.10  # the width of the bars
extra_offset = 0.06

fig, axs =  plt.subplots(ncols=3, sharey=True, figsize=(7.75,3))
colors = ['lightcoral', 'firebrick', 'palegoldenrod', 'goldenrod', 'lightskyblue', 'dodgerblue']
#colors = ['lightcoral', 'palegoldenrod', 'lightskyblue', 'firebrick', 'goldenrod', 'dodgerblue']
column_labels = [f'{ambiguity}, p({token})' for ambiguity in ['Ambiguous', 'GP', 'Non-GP'] for token in ['GP', 'non-GP']]
handles = []
for struct_mean, structure, ax in zip(means, ['NP/Z', 'NP/S', 'MV/RR'], axs):
    multiplier = 0
    offsets = [(i-1)* width + (i//2) * extra_offset for i in range(6)]
    #offsets = [(i-1)* width for i in range(6)]
    for i, (attribute, measurement) in enumerate(struct_mean.items()):
        rects = ax.bar(offsets[i], measurement, width, label=column_labels[i], color=colors[i], edgecolor='black')
        handles.append(rects)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(f'{structure}', y=-0.12)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
axs[0].set_ylabel('Probability')
# axs[1].set_xlabel('Garden Path Structure')
suptitle = fig.suptitle(f'Garden Path Continuation Probabilities ({model_name_mapping[model]})')

#handles = [handles[i] for i in [0,3,1,4,2,5]]
handles = handles[:6]
labels = [handle.get_label() for handle in handles]

leg = axs[1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3,
                    frameon=False)

# stylistic details
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.grid(False)

Path(f'behavioral-subplots').mkdir(exist_ok=True, parents=True)
plt.savefig(f'behavioral-subplots/{model}-behavioral-subplots.pdf', bbox_extra_artists=(leg,suptitle), bbox_inches='tight')
plt.show()
# %%
