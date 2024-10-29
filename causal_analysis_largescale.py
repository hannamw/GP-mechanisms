#%%
from typing import List, Dict, Iterable, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from typing import Literal
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer
from huggingface_hub import list_repo_files
from sae_lens import SAE

from dictionary_learning import dictionary

def get_performance(model:LanguageModel, tokens: torch.Tensor, indices,  dictionaries:Dict, features:Dict[str, List[Tuple]], use_inputs = None) -> torch.Tensor:
    """
    model: LanguageModel
    prompt: Iterable[str]
    dictionaries: {submodule: AutoEncoder}"""

    if use_inputs is None:
        use_inputs = {}
        for submodule_name, (submodule, ae) in dictionaries.items():
            if not model.config._name_or_path.startswith("google/"):
                use_inputs[submodule] = False
            elif not submodule_name.startswith("attn"):
                use_inputs[submodule] = False
            else:
                use_inputs[submodule] = True
    if model.config._name_or_path.startswith("google/"):
        model_out = model.lm_head
    elif model.config._name_or_path.startswith("EleutherAI/pythia"):
        model_out = model.embed_out

    with model.trace(tokens), torch.no_grad():
        for submodule_name, (submodule, ae) in dictionaries.items():                
            if len(features[submodule_name]) == 0:
                continue
                
            x = submodule.output if not use_inputs[submodule] else submodule.input
            if use_inputs[submodule]:
                x = x[0][0]
            elif type(x.shape) == tuple:
                x = x[0]
            f = ae.encode(x)
            x_hat = ae.decode(f)
            x_hat_orig = ae.decode(f)
            residual = x - x_hat_orig

            f_new = torch.clone(f)
            feature_pos, feature_list, feature_values = zip(*features[submodule_name])
            f_new[:, feature_pos, feature_list] = torch.tensor(feature_values, device='cuda').to(model.dtype)
            x_hat = ae.decode(f_new)
            if use_inputs[submodule]:
                submodule.input[0][0][:] = x_hat + residual
            elif type(submodule.output.shape) == tuple:
                submodule.output[0][:] = x_hat + residual
            else:
                submodule.output = x_hat + residual
        logits_saved = model_out.output.save()
    logits = logits_saved.value
    probs = torch.nn.functional.softmax(logits.squeeze(0), dim=-1)
    gp_indices, nongp_indices = indices
    gp_probs = probs[-1, gp_indices].sum().item()
    nongp_probs = probs[-1, nongp_indices].sum().item()   

    return gp_probs, nongp_probs


def submodule_name_to_submodule(model_name, submodule_name):
    if submodule_name == 'embed':
        if 'pythia' in model_name:
            return model.gpt_neox.embed_in
        elif 'gemma' in model_name:
            return model.model.embed_tokens
    submod_type, layer_idx = submodule_name.split("_")
    layer_idx = int(layer_idx)
    if 'llama' in model_name:
        if submod_type == "resid":
            return model.model.layers[layer_idx]
        elif submod_type == "attn":
            return model.model.layers[layer_idx].attn
        elif submod_type == "mlp":
            return model.model.layers[layer_idx].mlp
        else:
            raise ValueError(f"Unrecognized submodule type: {submod_type}")
    elif 'pythia' in model_name:
        if submod_type == "resid":
            return model.gpt_neox.layers[layer_idx]
        elif submod_type == "attn":
            return model.gpt_neox.layers[layer_idx].attention
        elif submod_type == "mlp":
            return model.gpt_neox.layers[layer_idx].mlp
        else:
            raise ValueError(f"Unrecognized submodule type: {submod_type}")
    elif 'gemma' in model_name:
        if submod_type == "resid":
            return model.model.layers[layer_idx]
        elif submod_type == "attn":
            return model.model.layers[layer_idx].self_attn.o_proj
        elif submod_type == "mlp":
            return model.model.layers[layer_idx].post_feedforward_layernorm
        else:
            raise ValueError(f"Unrecognized submodule type: {submod_type}")
    else:
        raise ValueError(f"Unrecognized model name: {model_name}")


def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
):
    if neurons:
        if submod_type != "attn":
            return dictionary.IdentityDict(2304)
        else:
            return dictionary.IdentityDict(2048)

    repo_id = "google/gemma-scope-2b-pt-" + (
        "res" if submod_type in ["embed", "resid"] else
        "att" if submod_type == "attn" else
        "mlp"
    )
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return dictionary.JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )

def load_autoencoder(model_name, submodule_name):
    if submodule_name == 'embed':
        # For Llama
        if 'llama' in model_name:
            ae_path = f"llama3-8b-saes/embed/ae_81920.pt"
            ae = dictionary.GatedAutoEncoder(4096, 32768).to("cuda")
        elif 'pythia' in model_name:
            ae_path = f"feature-circuits-gp/dictionaries/pythia-70m-deduped/embed/10_32768/ae.pt"
            ae = dictionary.AutoEncoder(512, 32768).to("cuda")
        elif 'gemma' in model_name:
            raise ValueError("Gemma does not have embedding SAEs")
    else:
        submod_type, layer_idx = submodule_name.split("_")
        # For Llama
        if 'llama' in model_name:
            ae_path = f"llama3-8b-saes/layer{layer_idx}/ae_81920.pt"
            ae = dictionary.GatedAutoEncoder(4096, 32768).to("cuda")
            ae.load_state_dict(torch.load(open(ae_path, "rb")))
            ae = ae.half()
        elif 'pythia' in model_name:
            ae_path = f"feature-circuits-gp/dictionaries/pythia-70m-deduped/{submod_type}_out_layer{layer_idx}/10_32768/ae.pt"
            ae = dictionary.AutoEncoder(512, 32768).to("cuda")
            ae.load_state_dict(torch.load(open(ae_path, "rb")))
            #ae = ae.half()
        elif 'gemma' in model_name:
            ae = load_gemma_sae(submod_type, int(layer_idx)).to("cuda")
            ae = ae.to(torch.bfloat16)
    return ae

def sentence_to_indices(s:str, tokenizer:AutoTokenizer):
    tokd = tokenizer(s, return_offsets_mapping=True)
    ss = s.split()
    start_index = len(' '.join(ss[:-3]))
    end_index = len(' '.join(ss[:-2]))
    verb_indices = []
    for i, (s,e) in enumerate(tokd.offset_mapping):
        if s >= end_index:
            break 
        
        if s >= start_index:
            verb_indices.append(i)
            
    np_indices = list(range(max(verb_indices) + 1, len(tokd.input_ids)))
    
    return tokd.input_ids, verb_indices, np_indices

#%%

if __name__ == '__main__':
    model_name ="EleutherAI/pythia-70m-deduped"
    #model_name = "google/gemma-2-2b"
    model_name_noslash = model_name.split('/')[-1]
    dataset_name = "data_csv/gp_same_len.csv"
    dtype = torch.bfloat16 if model_name.startswith("google/") else torch.float32
    high_value = 100.0 if 'gemma' in model_name else 2.0

    df = pd.read_csv('data_csv/ClassicGardenPathSet.csv')
    df = df[df['WordPosition'] == 1]
    sentences = defaultdict(list)
    for structure in ['NPZ', 'NPS']:
        structure_df = df[df['Type'] == f'{structure}_AMB']
        for sentence, critical_position in zip(structure_df['Sentence'], structure_df['CriticalPosition']):
            sentences[structure].append(' '.join(sentence.split()[:critical_position-1]))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LanguageModel(model_name, torch_dtype=dtype, device_map="cuda", dispatch=True)

    if 'llama' in model_name:
        all_submodule_names = []
    elif 'pythia' in model_name:
        all_submodule_names = ['embed', *(f'{module}_' + str(i) for i in range(6) for module in ['attn', 'mlp', 'resid'])]
    elif 'gemma' in model_name:
        all_submodule_names = [*(f'{module}_' + str(i) for i in range(26) for module in ['attn', 'mlp', 'resid'])]
    dictionaries = {submodule_name: (submodule_name_to_submodule(model_name, submodule_name), load_autoencoder(model_name, submodule_name)) for submodule_name in tqdm(all_submodule_names, desc="Loading submodules", total=len(all_submodule_names))}

    dict_size = dictionaries['resid_0'][1].dict_size

    baseline_probabilities = defaultdict(list)
    intervened_probabilities = defaultdict(list)
    random_probabilities = defaultdict(list)
    for condition in ['NPZ', 'NPS']:
        if condition == 'NPZ':
            gp_tokens = [',']
            post_tokens = [' was']
            upweight_categories = ['subject detector']
            downweight_categories = ['object detector']
        else:
            gp_tokens = ['.']
            post_tokens = [' was']
            upweight_categories = ['object detector', 'end of sentence detector']
            downweight_categories = ['subject detector', 'CP verb detector']

        gp_token_ids = [tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in gp_tokens]
        post_token_ids = [tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in post_tokens]
        indices = (gp_token_ids, post_token_ids)
        
        feature_df = pd.read_csv(f'results/{model_name_noslash}/{condition.lower()}_features.csv')
        feature_submodule_names, feature_idxs =  zip(*[feature.split('/') for feature in feature_df['Feature']])
        feature_df['submodule_name'] = feature_submodule_names
        feature_df['feature_idx'] = [int(x) for x in feature_idxs]
        
        annotated_features = defaultdict(list)
        for subname, idx in zip(feature_df['submodule_name'], feature_df['feature_idx']):
            annotated_features[subname].append(idx)
            
        mapping = defaultdict(dict)
        for subname, idxs in annotated_features.items():
            candidates = torch.randperm(dict_size)
            i = 0
            for j in idxs:
                while candidates[i].item() in annotated_features[subname]:
                    i += 1
                mapping[subname][j] = candidates[i].item()
                
        for s in tqdm(sentences[condition]):
            tokens, verb_indices, np_indices = sentence_to_indices(s, tokenizer)
            tokens = torch.tensor(tokens, device='cuda').unsqueeze(0)
            
            highlevel_features = defaultdict(list)
            random_features = defaultdict(list)
            for position, feature, category in zip(feature_df['Position'], feature_df['Feature'], feature_df['Category']):
                submodule_name, feature_idx = feature.split('/')
                feature_idx = int(feature_idx)
                random_feature_idx = mapping[submodule_name][feature_idx]
                if category == 'end of clause detector':
                    for verb_idx in verb_indices:
                        highlevel_features[submodule_name].append((verb_idx, feature_idx, high_value))
                        random_features[submodule_name].append((verb_idx, random_feature_idx, high_value))
                        
                    for noun_idx in np_indices:
                        highlevel_features[submodule_name].append((noun_idx, feature_idx, 0.0))
                        random_features[submodule_name].append((noun_idx, random_feature_idx, 0.0))
                        
                elif category in downweight_categories:
                    if category == 'subject detector' or category == 'object detector':
                        for noun_idx in np_indices:
                            highlevel_features[submodule_name].append((noun_idx, feature_idx, 0.0))
                            random_features[submodule_name].append((noun_idx, random_feature_idx, 0.0))
                            
                    if category == 'CP verb detector':
                        for verb_idx in verb_indices:
                            highlevel_features[submodule_name].append((verb_idx, feature_idx, 0.0))
                            random_features[submodule_name].append((verb_idx, random_feature_idx, 0.0))
                            
                elif category in upweight_categories:
                    if category == 'subject detector' or category == 'object detector' or category == 'end of sentence detector':
                        for noun_idx in np_indices:
                            highlevel_features[submodule_name].append((noun_idx, feature_idx, high_value))
                            random_features[submodule_name].append((noun_idx, random_feature_idx, high_value))
                
            intervened_gp, intervened_nongp = get_performance(model, tokens, indices, dictionaries, highlevel_features)
            random_gp, random_nongp = get_performance(model, tokens, indices, dictionaries, random_features)
            baseline_gp, baseline_nongp = get_performance(model, tokens, indices, dictionaries, defaultdict(list))
            
            intervened_probabilities[condition].append((intervened_gp, intervened_nongp))
            random_probabilities[condition].append((random_gp, random_nongp))
            baseline_probabilities[condition].append((baseline_gp, baseline_nongp))
        
all_probabilities = {'intervened': intervened_probabilities, 'baseline': baseline_probabilities, 'random': random_probabilities}
Path(f'results/{model_name_noslash}').mkdir(exist_ok=True, parents=True)
torch.save(all_probabilities, f'results/{model_name_noslash}/causal_probabilities_largescale.pt')
    

# %%
