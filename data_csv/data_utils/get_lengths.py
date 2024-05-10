#%%
import pandas as pd
from transformers import AutoTokenizer

model_name = 'EleutherAI/pythia-70m'
tokenizer = AutoTokenizer.from_pretrained(model_name)
#%%
df = pd.read_csv('garden_path_sentences_trimmed.csv')

for col in ['ambiguous', 'gp', 'post']:
    colname = f'sentence_{col}'
    lens = [len(tokenizer(s)['input_ids']) for s in df[colname]]
    df[f'{colname}_len'] = lens 

df.to_csv('garden_path_sentences_trimmed_lens.csv')
# %%
df = pd.read_csv('garden_path_sentences_same_len.csv')
for col in ['ambiguous', 'gp', 'post']:
    colname = f'sentence_{col}'
    lens = [len(tokenizer(s)['input_ids']) for s in df[colname]]
    df[f'{colname}_len'] = lens 

df.to_csv('garden_path_sentences_same_len.csv')
# %%
conditions = set(df['condition'].tolist())
for condition in conditions:
    condition_df = df[df['condition']==condition]
    print(((condition_df['sentence_ambiguous_len'] == condition_df['sentence_gp_len'] ) & (condition_df['sentence_gp_len'] == condition_df['sentence_post_len'])).all())
# %%
