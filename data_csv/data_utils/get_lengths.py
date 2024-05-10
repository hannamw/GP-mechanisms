#%%
import pandas as pd

from transformers import AutoTokenizer
#%%
df = pd.read_csv('garden_path_sentences_trimmed.csv')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
for col in ['ambiguous', 'gp', 'post']:
    colname = f'sentence_{col}'
    lens = [len(tokenizer(s)['input_ids']) for s in df[colname]]
    df[f'{colname}_len'] = lens 

df.to_csv('garden_path_sentences_trimmed_lens.csv')
# %%
df = pd.read_csv('garden_path_sentences_same_len.csv')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
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
