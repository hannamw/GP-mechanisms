#%%
from functools import partial
from pathlib import Path

import pandas as pd 

import torch 
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer

from eap.graph import Graph 
from eap.attribute import attribute
from eap.evaluate_graph import evaluate_graph, evaluate_baseline

from utils import GPDataset, kl_div, collate_fn, logit_diff
#%%
df = pd.read_csv('garden_path_sentences_trimmed.csv')
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

image_path = Path(f'images/{model_name_noslash}/')
graph_path = Path(f'graphs/{model_name_noslash}/')
image_path.mkdir(exist_ok=True, parents=True)
graph_path.mkdir(exist_ok=True, parents=True)
#%%
cols = {'sentence_ambiguous' , 'sentence_post', 'sentence_gp'}

#%%
for cln in cols:
    for cor in cols:
        if cln == cor:
            continue
        batch_size = 8
        ds = GPDataset(model, df, clean=cln, corrupted=cor)
        dataloader = DataLoader(ds, collate_fn=collate_fn, batch_size=batch_size)
        
        baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean()
        #print(baseline)
        corrupted_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), run_corrupted=True).mean()
        #print(corrupted_baseline)
        
        graph = Graph.from_model(model)
        attribute(model,graph, dataloader, kl_div, integrated_gradients=5)
        
        n = int(len(graph.edges) * 0.05)
        graph.apply_greedy(n, absolute=True)
        graph.prune_dead_nodes()
        gz = graph.to_graphviz()
        gz.draw(f'images/{model_name_noslash}/{ds.name()}.png', prog='dot')
        
        performance = evaluate_graph(model, graph, dataloader, partial(logit_diff, mean=False, loss=False)).mean()
        normalized_performance = (performance - corrupted_baseline) / (baseline - corrupted_baseline)
        print(cln, cor)
        print(performance)
        print(normalized_performance)
        graph.to_json(f'graphs/{model_name_noslash}/{ds.name()}.json')


# %%
