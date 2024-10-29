#%%
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from curlyBrace import curlyBrace

model = 'pythia-70m-deduped'
model = 'gemma-2-2b'
# maybe we want to change the display names of each model
model_name_mapping = {'pythia-70m-deduped': 'Pythia-70m', 'gemma-2-2b': 'Gemma-2-2B'}
df = pd.read_csv(f'../results/{model}/gp_with_probs.csv')
sns.set_theme(style='whitegrid')

means = [{f'{ambiguity}': [df[df['condition'] == category][f'{ambiguity}_punct_prob'].mean() - df[df['condition'] == category][f'{ambiguity}_was_prob'].mean()] for ambiguity in ['ambiguous','gp','post']} for category in ['NPZ', 'NPS', 'MVRR']]
all_means = []
#means = {f'{ambiguity}_{token}': [df[df['condition'] == category][f'{ambiguity}_{token}_prob'].mean() for category in ['NPZ', 'NPS', 'MVRR']] for token in ['punct', 'was'] for ambiguity in ['ambiguous','gp','post']}

categories = ['NP/Z', 'NP/S', 'MV/RR']
x = np.array([0])  # the label locations
width = 0.10  # the width of the bars
extra_offset = 0.06

fig, axs =  plt.subplots(ncols=3, sharey=True, figsize=(7.75,3))
#colors = ['firebrick', 'goldenrod', 'dodgerblue']
colors = ['lightcoral', 'palegoldenrod', 'lightskyblue']
#colors = ['lightcoral', 'palegoldenrod', 'lightskyblue', 'firebrick', 'goldenrod', 'dodgerblue']
column_labels = [f'{ambiguity}' for ambiguity in ['Ambiguous', 'GP', 'Non-GP']]
handles = []
for struct_mean, structure, ax in zip(means, ['NP/Z', 'NP/S', 'MV/RR'], axs):
    multiplier = 0
    offsets = [(i-1)* width + (i) * extra_offset for i in range(3)]
    #offsets = [(i-1)* width for i in range(6)]
    for i, (_, measurement) in enumerate(struct_mean.items()):
        all_means.append(measurement[0])
        rects = ax.bar(offsets[i], measurement, width, label=column_labels[i], color=colors[i], edgecolor='black')
        handles.append(rects)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(f'{structure}', y=-0.12)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
axs[0].set_ylabel('p(GP)-p(non-GP)')
# axs[1].set_xlabel('Garden Path Structure')
suptitle = fig.suptitle(f'Garden Path Continuation Probabilities ({model_name_mapping[model]})')

#handles = [handles[i] for i in [0,3,1,4,2,5]]
handles = handles[:3]
labels = [handle.get_label() for handle in handles]

leg = axs[1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, title="Input Sentence Type")

# stylistic details
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.grid(False)
    ax.axhline(0, color='black', linewidth=1.0)
    
x = """
axs[0].annotate('GP', xy=(-0.25, 0.75), xytext=(-0.6, 0.79), xycoords='axes fraction', ha='left', va='top',
            bbox=dict(boxstyle='square', fc='white', color='white'),
            arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=0.35', lw=2.0, color='k'))
axs[0].annotate('Non\n-GP', xy=(-0.25, 0.22), xytext=(-0.68, 0.3), xycoords='axes fraction', ha='left', va='top',
            bbox=dict(boxstyle='square', fc='white', color='white'),
            arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=0.35', lw=2.0, color='k'))
"""
extra_artists1 = curlyBrace(fig, axs[-1], (0.3, max(all_means)), (0.3,0.01), str_text='GP', color='black', clip_on=False)[-1]
extra_artists2 = curlyBrace(fig, axs[-1], (0.3, -0.01), (0.3,min(all_means)), str_text='non-GP', color='black', clip_on=False)[-1]
axs[-1].set_xlim(axs[-2].get_xlim())

Path(f'behavioral-subplots-difference').mkdir(exist_ok=True, parents=True)
plt.savefig(f'behavioral-subplots-difference/{model}-behavioral-subplots-difference.pdf', bbox_extra_artists=(leg,suptitle, *extra_artists1, *extra_artists2), bbox_inches='tight')
plt.show()
# %%
