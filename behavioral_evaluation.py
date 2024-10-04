#%%
import pandas as pd 
from transformer_lens import HookedTransformer
import torch
#%%
df = pd.read_csv('data_csv/gp_same_len.csv')
model_name = 'google/gemma-2-2b'
save_name = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name)
#%%

post_token = ' was'
post_token_id = model.tokenizer(post_token, add_special_tokens=False)['input_ids'][0]
for column in ['sentence_ambiguous','sentence_gp','sentence_post']:
    ambiguity = column.split('_')[1]
    gp_probs = []
    post_probs = []
    for sentence, condition in zip(df[column], df['condition']):
        gp_token = ',' if 'NPZ' in condition  else '.'
        gp_token_id = model.tokenizer(gp_token, add_special_tokens=False)['input_ids'][0]
    
        probs = torch.softmax(model(sentence).squeeze(0)[-1], -1)
        gp_prob = probs[gp_token_id]
        post_prob = probs[post_token_id]
        
        gp_probs.append(gp_prob.item())
        post_probs.append(post_prob.item())
        
    df[f'{ambiguity}_punct_prob'] = gp_probs
    df[f'{ambiguity}_was_prob'] = post_probs

df.to_csv(f'results/{save_name}/gp_with_probs.csv', index=False)
# %%
d = {'condition': [], '% post': [], '% gp':[]}
for ambiguity in ['ambiguous','gp','post']:
    d['condition'].append(f'{ambiguity}_all')
    df_no_comma = df[df['condition'] != 'NPZ_comma']
    post_percent = (df_no_comma[f'{ambiguity}_was_prob'] > df_no_comma[f'{ambiguity}_punct_prob']).mean()
    d['% post'].append(post_percent)
    d['% gp'].append(1 - post_percent)
    
    for condition in ['NPZ', 'NPS', 'MVRR', 'NPZ_comma']:
        d['condition'].append(f'{ambiguity}_{condition}')
        filtered_df = df[df['condition'] == condition]
        post_percent = (filtered_df[f'{ambiguity}_was_prob'] > filtered_df[f'{ambiguity}_punct_prob']).mean()
        d['% post'].append(post_percent)
        d['% gp'].append(1 - post_percent)

df2 = pd.DataFrame(d)
df2.to_csv(f'results/{save_name}/behavioral_summary.csv', index=False)
# %%
