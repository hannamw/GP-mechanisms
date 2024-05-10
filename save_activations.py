#%%
from pathlib import Path
import pandas as pd 
import torch
from transformer_lens import HookedTransformer 
#%%
model_name = 'gpt2'
save_name = model_name.split('/')[-1]
p = Path(f"activations/{save_name}")
p.mkdir(exist_ok=True, parents=True)
model = HookedTransformer.from_pretrained(model_name)
df = pd.read_csv("data_csv/gp_same_len.csv")
activation_points = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
#%%
for column in ['sentence_ambiguous','sentence_gp','sentence_post']:
    for condition in ['NPS', 'NPZ', 'MVRR']:
        sentences = df[df['condition']==condition][column].tolist()
        with torch.inference_mode():
            _, cache = model.run_with_cache(sentences)
        for activation_point in activation_points:
            activations = cache[activation_point].cpu()
            torch.save(activations, p / f"{column}-{condition}-{activation_point}.pt")    

# %%
