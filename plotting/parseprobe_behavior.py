#%%
from pathlib import Path 
import torch

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
#%%
model_name = 'EleutherAI/pythia-70m-deduped'
model_name_noslash = model_name.split('/')[-1]
# %%
# NOTE: In reality, dimensions 0, 1, and 2 of the probe probs correspond to GEN, LEFT-ARC, and RIGHT-ARC, respectively.
# But, this is a little confusing: the proper order to feed the representations into the probe is [last, verb], but this is the opposite of how they appear in the sentence! This has to do with the overall parsing algorithm in the context of which the probe is used, I think
# So, LEFT-ARC means drawing an arc from the verb to the last token, and RIGHT-ARC means drawing an arc from the last token to the verb, the opposite of what you'd expect.

probs = torch.load(f'../results/{model_name_noslash}/parse_probe/probe_probs.pt')
fig, ax  = plt.subplots(figsize=(9, 3))

for key, data in probs.items():
    condition, sentence_type = key.split('-')
    if condition == 'NPZ':
        if sentence_type == 'sentence_gp':
            color = 'goldenrod'
            label = 'NP/Z GP'
        elif sentence_type == 'sentence_ambiguous':
            color = 'blue'
            label = 'NP/Z Ambiguous'
            continue
        else:
            color = 'darkolivegreen'
            label = 'NP/Z Non-GP'
    else:
        if sentence_type == 'sentence_gp':
            color = 'firebrick'
            label = 'NP/S GP'
        elif sentence_type == 'sentence_ambiguous':
            color = 'orange'
            label = 'NP/S Ambiguous'
            continue
        else:
            color = 'dodgerblue'
            label = 'NP/S Non-GP'
    #color = 'blue' if condition == 'NPZ' else 'orange'
    condition_data = torch.stack([data[layer] for layer in ['embed', *(f'resid_{i}' for i in range(6))]], dim=0)
    mean_data = condition_data.mean(1).numpy()
    for (i, action, linetype) in zip([0,2], ['GEN', 'LEFT-ARC'], ['-', '--']):
    #for i, (action, linetype) in enumerate(zip(['GEN', 'RIGHT-ARC', 'LEFT-ARC'], ['-', '--', ':'])):
        ax.plot(mean_data[:, i], label=f'{label} {action}', linestyle=linetype, color=color)
        
ax.set_xlabel('Layer')
ax.set_ylabel('Probability')
ax.set_xticks(list(range(7)), ['embeds', *(str(x) for x in range(6))])
ax.set_title('Probe Action Probabilities')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
#zipped = list(zip(handles, labels))
#handles, labels = zip(*[zipped[i] for i in [0,3,1,4,2,5]])
leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fancybox=True)

fig.show()
Path(f'parse-probe-behavioral-comparison').mkdir(exist_ok=True)
fig.savefig(f'parse-probe-behavioral-comparison/{model_name_noslash}-parse-probe-behavioral-comparison.png', bbox_extra_artists=(leg,),bbox_inches='tight')
fig.savefig(f'parse-probe-behavioral-comparison/{model_name_noslash}-parse-probe-behavioral-comparison.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')
# %%
fig, ax  = plt.subplots(figsize=(6, 2.3))

for key, data in probs.items():
    condition, sentence_type = key.split('-')
    if sentence_type != 'sentence_ambiguous':
        continue 
    color = 'blue' if condition == 'NPZ' else 'orange'
    label = 'NP/Z' if condition == 'NPZ' else 'NP/S'
    condition_data = torch.stack([data[layer] for layer in ['embed', *(f'resid_{i}' for i in range(6))]], dim=0)
    mean_data = condition_data.mean(1).numpy()
    for i, (action, linetype) in enumerate(zip(['GEN', 'RIGHT-ARC', 'LEFT-ARC'], ['-', '--', ':'])):
        ax.plot(mean_data[:, i], label=f'{label} {action}', linestyle=linetype, color=color)
        
ax.set_xlabel('Layer')
ax.set_ylabel('Probability')
ax.set_xticks(list(range(7)), ['embeds', *(str(x) for x in range(6))])
ax.set_title('Probe Action Probabilities')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
zipped = list(zip(handles, labels))
handles, labels = zip(*[zipped[i] for i in [0,3,1,4,2,5]])
leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fancybox=True)

fig.show()
Path(f'parse-probe-behavioral').mkdir(exist_ok=True)
fig.savefig(f'parse-probe-behavioral/{model_name_noslash}-parse-probe-behavioral.pdf', bbox_extra_artists=(leg,),bbox_inches='tight')
fig.savefig(f'parse-probe-behavioral/{model_name_noslash}-parse-probe-behavioral.png', bbox_extra_artists=(leg,),bbox_inches='tight')
# %%
