#%%
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from einops import rearrange
import torch 
from concept_erasure import LeaceEraser
from transformer_lens import HookedTransformer
#%%
model_name = 'gpt2'
save_name = model_name.split('/')[-1]
p = Path(f"activations/{save_name}")
model = HookedTransformer.from_pretrained(model_name)
df = pd.read_csv('data_csv/gp_same_len.csv')
condition = 'NPZ'
df = df[df['condition'] == condition]

gp_token = ',' if 'NPZ' in condition  else '.'
gp_token_id = model.tokenizer(gp_token, add_special_tokens=False)['input_ids'][0]
        
post_token = ' was'
post_token_id = model.tokenizer(post_token, add_special_tokens=False)['input_ids'][0]

sentences = {column:df[column].tolist() for column in ['sentence_ambiguous','sentence_gp','sentence_post']}
activation_points = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
#%%
d = defaultdict(list)
for activation_point in activation_points:
    d['activation_point'].append(activation_point)
    print(activation_point)
    acts_ambiguous = torch.load(p / f"sentence_ambiguous-{condition}-{activation_point}.pt")[:, -3:]
    acts_gp = torch.load(p / f"sentence_gp-{condition}-{activation_point}.pt")[:, -3:]
    acts_post = torch.load(p / f"sentence_post-{condition}-{activation_point}.pt")[:, -3:]
    
    ambiguous_x = rearrange(acts_ambiguous, 'b s d -> (b s) d')
    gp_x = rearrange(acts_gp, 'b s d -> (b s) d')
    post_x = rearrange(acts_post, 'b s d -> (b s) d')
    
    X = torch.cat([gp_x, post_x], dim=0)
    y = torch.ones(X.shape[0])
    y[:len(y)//2] = 0
    
    clf = LogisticRegression(max_iter=1000).fit(X.numpy(), y.numpy())
    preds = clf.predict(X)
    accuracy = (preds == y.numpy()).mean()
    print(f"Accuracy: {accuracy:.2f}")
    d['accuracy'].append(accuracy)
    ambiguous_preds = clf.predict(ambiguous_x)
    mean_ambiguous_prediction = ambiguous_preds.mean()
    print(f"Mean ambiguous prediction: {mean_ambiguous_prediction:.2f}")
    d['mean_ambiguous_prediction'].append(mean_ambiguous_prediction)
    
    eraser = LeaceEraser.fit(X,y)
    X_ = eraser(X)

    # But learns nothing after
    null_lr = LogisticRegression(max_iter=1000, tol=0.0).fit(X_.numpy(), y.numpy())
    null_preds = null_lr.predict(X_.numpy())
    null_accuracy = (null_preds == y.numpy()).mean()
    print(f"Null accuracy: {null_accuracy:.2f}")
    d['null_accuracy'].append(null_accuracy)
    
    with torch.inference_mode():
        gp_probs = torch.softmax(model(sentences['sentence_gp'])[:, -1], dim=-1)
        post_probs = torch.softmax(model(sentences['sentence_post'])[:, -1], dim=-1)
        ambiguous_probs = torch.softmax(model(sentences['sentence_ambiguous'])[:, -1], dim=-1)
        
    gp_pref = (gp_probs[:, gp_token_id] > gp_probs[:, post_token_id]).float().mean().cpu().item()
    post_pref = (post_probs[:, post_token_id] > post_probs[:, gp_token_id]).float().mean().cpu().item()
    ambiguous_pref = (ambiguous_probs[:, post_token_id] > ambiguous_probs[:, gp_token_id]).float().mean().cpu().item()
    print(f"GP preference: {gp_pref:.2f}")
    print(f"Post preference: {post_pref:.2f}")
    print(f"Ambiguous preference: {ambiguous_pref:.2f}")
    d['gp_pref'].append(gp_pref)
    d['post_pref'].append(post_pref)
    d['ambiguous_pref'].append(ambiguous_pref)
        
    def leace_hook(x, hook):
        eraser_input = rearrange(x[:, -3:], 'b s d -> (b s) d').cpu()
        eraser_output = eraser(eraser_input)
        new_x_values = rearrange(eraser_output, '(b s) d -> b s d', s=3)
        x[:, -3:] = new_x_values.cuda()
        return x.cuda()
        
    with model.hooks([(activation_point, leace_hook)]):
        gp_probs_leace = torch.softmax(model(sentences['sentence_gp'])[:, -1], dim=-1)
        post_probs_leace = torch.softmax(model(sentences['sentence_post'])[:, -1], dim=-1)
        ambiguous_probs_leace = torch.softmax(model(sentences['sentence_ambiguous'])[:, -1], dim=-1)
        
    gp_pref_leace = (gp_probs_leace[:, gp_token_id] > gp_probs_leace[:, post_token_id]).float().mean().cpu().item()
    post_pref_leace = (post_probs_leace[:, post_token_id] > post_probs_leace[:, gp_token_id]).float().mean().cpu().item()
    ambiguous_pref_leace = (ambiguous_probs_leace[:, post_token_id] > ambiguous_probs_leace[:, gp_token_id]).float().mean().cpu().item()
    
    print(f"GP preference after erasure: {gp_pref_leace:.2f}")
    print(f"Post preference after erasure: {post_pref_leace:.2f}")
    print(f"Ambiguous preference after erasure: {ambiguous_pref_leace:.2f}")
    d['gp_pref_leace'].append(gp_pref_leace)
    d['post_pref_leace'].append(post_pref_leace)
    d['ambiguous_pref_leace'].append(ambiguous_pref_leace)
    

df = pd.DataFrame(d)
print(condition)
df
# %%
