#%%
from pathlib import Path
import pandas as pd 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
#%%
df = pd.read_csv('data_csv/gp_same_len.csv')
model_name = 'EleutherAI/pythia-70m-deduped'
save_name = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.cuda()
#%%
for column in ['sentence_ambiguous','sentence_gp','sentence_post']:
    ambiguity = column.split('_')[1]
    gp_probs = []
    post_probs = []
    for sentence, condition in zip(df[column], df['condition']):
        if condition == 'NPZ':
            gp_tokens = [',']
            # post_tokens = [' was',  ' had', ' did', ' would', ' will', ' should', ' might']
            post_tokens = [' was']
        elif condition == 'NPS':
            gp_tokens = ['.']
            # post_tokens = [' was',  ' had', ' did', ' would', ' will', ' should', ' might']
            post_tokens = [' was']
        elif condition == 'MVRR':
            gp_tokens = ['.']
            # post_tokens = [' was',  ' had', ' did', ' would', ' will', ' should', ' might']
            post_tokens = [' was']
        else:
            raise ValueError(f'Invalid condition: {condition}')

        gp_token_ids = [tokenizer(np, add_special_tokens=False)['input_ids'][0] for np in gp_tokens]
        post_token_ids = [tokenizer(z, add_special_tokens=False)['input_ids'][0] for z in post_tokens]
    
        tokens = tokenizer(sentence, return_tensors='pt')['input_ids'].cuda()
        probs = torch.softmax(model(tokens).logits.squeeze(0)[-1], -1)
        gp_prob = probs[gp_token_ids].sum()
        post_prob = probs[post_token_ids].sum()
        
        gp_probs.append(gp_prob.item())
        post_probs.append(post_prob.item())
        
    df[f'{ambiguity}_punct_prob'] = gp_probs
    df[f'{ambiguity}_was_prob'] = post_probs

p = Path('results') / save_name
p.mkdir(exist_ok=True, parents=True)
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
