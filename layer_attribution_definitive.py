#%%
import importlib  
import sys
sys.path.append('feature-circuits-gp')

import torch
from torch import optim
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel

import attribution
import dictionary_learning as dl
#%%
DEVICE='cuda'
model_name = 'EleutherAI/pythia-70m-deduped'
model_name_noslash = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LanguageModel(model_name, device_map='auto')
model.eval()

df = pd.read_csv("data_csv/gp_same_len.csv")

scores = {condition: {} for condition in ['NPZ', 'NPS', 'MVRR']}

#%%
for condition in ['NPZ', 'NPS', 'MVRR']:
    if condition == 'NPZ':
        gp_tokens = [',']
        post_tokens = [' was',  ' had', ' did', ' would', ' will', ' should', ' might']
    elif condition == 'NPS':
        gp_tokens = ['.']
        post_tokens = [' was',  ' had', ' did', ' would', ' will', ' should', ' might']
    elif condition == 'MVRR':
        gp_tokens = ['.', ' to']
        post_tokens = [' was',  ' had', ' did', ' would', ' will', ' should', ' might']
    else:
        raise ValueError(f'Invalid condition: {condition}')

    gp_token_ids = [tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in gp_tokens]
    post_token_ids = [tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in post_tokens]

    def metric_fn(model, labels=None):
        logits = model.embed_out.output
        probs = torch.softmax(logits, dim=-1)
        
        prob_diff = probs[:, -1, post_token_ids].sum(-1) - probs[:, -1, gp_token_ids].sum(-1)
        return -prob_diff.mean()


    filtered_df = df[df['condition'] == condition]
    tokens = tokenizer(filtered_df['sentence_ambiguous'].tolist(), return_tensors='pt').to('cuda').input_ids
    tokens = torch.cat((torch.full((tokens.size(0),1), tokenizer.bos_token_id, device='cuda'), tokens) , dim=-1)

    for layer in ['embed', *range(6)]:
        if layer == 'embed':
            submodules = [model.gpt_neox.embed_in]
            dictionaries = {model.gpt_neox.embed_in: dl.dictionary.AutoEncoder(512, 32768).to("cuda")}
            state_dict = torch.load(f'feature-circuits-gp/dictionaries/pythia-70m-deduped/embed/10_32768/ae.pt')
            dictionaries[model.gpt_neox.embed_in].load_state_dict(state_dict)        
        else:
            submodules = [model.gpt_neox.layers[layer]]
            dictionaries = {model.gpt_neox.layers[layer]: dl.dictionary.AutoEncoder(512, 32768).to("cuda")}
            state_dict = torch.load(f'feature-circuits-gp/dictionaries/pythia-70m-deduped/resid_out_layer{layer}/10_32768/ae.pt')
            dictionaries[model.gpt_neox.layers[layer]].load_state_dict(state_dict)

        effects, _, _, _ = attribution.patching_effect(
            tokens,
            None,
            model,
            submodules,
            dictionaries,
            metric_fn,
            metric_kwargs=dict(),
            method='ig'
        )
        
        mean_effects = list(effects.values())[0].act.mean(0)
        scores[condition][layer] = mean_effects.cpu()
#%%
torch.save(scores, f'results/{model_name_noslash}/feature_scores.pt')
#%%
top_features = {condition: {layer: scores.abs().topk(20) for layer, scores in layer_scores.items()} for condition, layer_scores in scores.items()}
torch.save(top_features, f'results/{model_name_noslash}/topk_features.pt')
# %%
