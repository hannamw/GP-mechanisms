#%%
import jsonlines
import pandas as pd
from transformer_lens import HookedTransformer

from utils import GPDataset
#%%
model_name = 'EleutherAI/pythia-160m'
model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
#%%
df = pd.read_csv('garden_path_sentences_trimmed.csv')
cols = ['sentence_ambiguous' , 'sentence_post', 'sentence_gp']

for cln in cols:
    for cor in cols:
        if cln == cor:
            continue
        for subclass in ['NPS', 'NPZ', 'MVRR']:
            subclass_df = df[df['condition']==f'{subclass}_UAMB'].reset_index()
            ds = GPDataset(model, subclass_df, clean=cln, corrupted=cor, token_id_labels=False)

            dicts = [{"clean_prefix": ds[i][0], "patch_prefix": ds[i][1], "clean_answer": ds[i][2][0], "patch_answer": ds[i][2][1], "case": subclass} for i in range(len(ds))]

            cln_short = cln.split('_')[-1]
            cor_short = cor.split('_')[-1]
            with open(f'feature-circuits-gp/data/{cln_short}_{cor_short}_{subclass}.json', 'w') as f:
                with jsonlines.Writer(f) as writer:
                    for d in dicts:
                        writer.write(d)
# %%
