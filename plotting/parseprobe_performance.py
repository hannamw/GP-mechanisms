#%%
from pathlib import Path 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
#%%
model_name = 'EleutherAI/pythia-70m-deduped'
model_name_noslash = model_name.split('/')[-1]

fig, ax  = plt.subplots(figsize=(6, 3))

data = [pd.read_csv(f'../results/{model_name_noslash}/parse_probe/performance/results_{model_name_noslash}_layer_{i}_StackActionProbe_beamsearch.csv') for i in range(7)]
uas = [df[df['metric'] == 'uas_beamsearch']['value'].mean() for df in data]
uuas = [df[df['metric'] == 'uuas_beamsearch']['value'].mean() for df in data]

ax.plot(uas, label='UAS', color='blue')
ax.plot(uuas, label='UUAS', color='orange')
        
ax.set_xlabel('Layer')
ax.set_ylabel('(U)UAS')
ax.set_ylim(0, 1)
ax.set_xticks(list(range(7)), ['embeds', *(str(x) for x in range(6))])
ax.set_title('Parse Probe Performance')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fancybox=True)

fig.show()
Path(f'parse-probe-performance').mkdir(exist_ok=True)
fig.savefig(f'parse-probe-performance/{model_name_noslash}-parse-probe-performance.pdf', bbox_extra_artists=(leg,),bbox_inches='tight')
fig.savefig(f'parse-probe-performance/{model_name_noslash}-parse-probe-performance.png', bbox_extra_artists=(leg,),bbox_inches='tight')
# %%
