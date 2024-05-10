#%%
import pandas as pd 
import torch
from transformer_lens import HookedTransformer 
#%%
model = HookedTransformer.from_pretrained('EleutherAI/pythia-70m')
df = pd.read_csv("data_csv/garden_path_sentences_same_len.csv")
activation_points = [f'blocks.{i}.hook_resid_post' for i in range(6)]
#%%
for column in ['sentence_ambiguous','sentence_gp','sentence_post']:
    for condition in ['NPS', 'NPZ', 'MVRR']:
        sentences = df[df['condition']==condition][column].tolist()
        _, cache = model.run_with_cache(sentences)
        for activation_point in activation_points:
            activations = cache[activation_point].cpu()
            torch.save(activations, f"activations/{column}-{condition}-{activation_point}.pt")    

# %%
