#%%
from typing import List, Dict, Iterable, Tuple
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer
from sae_lens import SAE
from dictionary_learning import dictionary
            

def get_performance(model:LanguageModel, prompts: Iterable[str], indices,  dictionaries:Dict, features:Dict[str, List[Tuple]]) -> torch.Tensor:
    """
    model: LanguageModel
    prompt: Iterable[str]
    dictionaries: {submodule: AutoEncoder}"""

    gp_probs, nongp_probs = [], []
    for prompt in prompts:
        with model.trace(prompt), torch.no_grad():
            for submodule_name, (submodule, ae) in dictionaries.items():                
                if len(features[submodule_name]) == 0:
                    continue
                    
                x = submodule.output
                if type(x.shape) == tuple:
                    x = x[0]
                f = ae.encode(x)
                x_hat = ae.decode(f)
                x_hat_orig = ae.decode(f)
                residual = x - x_hat_orig

                f_new = torch.clone(f)
                feature_pos, feature_list, feature_values = zip(*features[submodule_name])
                f_new[:, feature_pos, feature_list] = torch.tensor(feature_values, device='cuda').half()
                x_hat = ae.decode(f_new)
                if type(submodule.output.shape) == tuple:
                    submodule.output[0][:] = x_hat + residual
                else:
                    submodule.output = x_hat + residual
            logits_saved = model.embed_out.output.save()
        logits = logits_saved.value
        probs = torch.nn.functional.softmax(logits.squeeze(0), dim=-1)
        gp_indices, nongp_indices = indices
        gp_probs.append(probs[-1, gp_indices].sum().item())
        nongp_probs.append(probs[-1, nongp_indices].sum().item())       

    return (torch.tensor(gp_probs), torch.tensor(nongp_probs))


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
            return model.model.layers[layer_idx].self_attn
        elif submod_type == "mlp":
            return model.model.layers[layer_idx].mlp
        else:
            raise ValueError(f"Unrecognized submodule type: {submod_type}")
    else:
        raise ValueError(f"Unrecognized model name: {model_name}")


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
        elif 'pythia' in model_name:
            ae_path = f"feature-circuits-gp/dictionaries/pythia-70m-deduped/{submod_type}_out_layer{layer_idx}/10_32768/ae.pt"
            ae = dictionary.AutoEncoder(512, 32768).to("cuda")
        elif 'gemma' in model_name:
            ae, cfg_dict, sparsity = SAE.from_pretrained(
            release = f"gemma-scope-2b-pt-{submod_type[:3]}-canonical",
            sae_id = f"layer_{layer_idx}/width_16k/canonical",
        )
    ae.load_state_dict(torch.load(open(ae_path, "rb")))
    ae = ae.half()
    return ae

if __name__ == '__main__':
    model_name ="EleutherAI/pythia-70m-deduped"
    model_name_noslash = model_name.split('/')[-1]
    dataset_name = "data_csv/gp_same_len.csv"
    
    upweight_categories = ['subject detector']
    downweight_categories = ['object detector']

    df = pd.read_csv(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False)
    model = LanguageModel(model_name, torch_dtype=torch.float16, device_map="cuda")

    if 'llama' in model_name:
        all_submodule_names = []
    elif 'pythia' in model_name:
        all_submodule_names = ['embed', *(f'{module}_' + str(i) for i in range(6) for module in ['attn', 'mlp', 'resid'])]
    elif 'gemma' in model_name:
        all_submodule_names = [*(f'{module}_' + str(i) for i in range(26) for module in ['attn', 'mlp', 'resid'])]
    dictionaries = {submodule_name: (submodule_name_to_submodule(model_name, submodule_name), load_autoencoder(model_name, submodule_name)) for submodule_name in all_submodule_names}

    baseline_probabilities = {}
    intervened_probabilities = {}
    for condition in ['NPZ']:#, 'NPS']:
        if condition == 'NPZ':
            gp_tokens = [',']
            post_tokens = [' was']
        else:
            gp_tokens = ['.']
            post_tokens = [' was']

        gp_token_ids = [tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in gp_tokens]
        post_token_ids = [tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in post_tokens]
        indices = (gp_token_ids, post_token_ids)
        condition_df = df[df['condition'] == condition]
        baseline_gp, baseline_nongp = get_performance(model, condition_df['sentence_ambiguous'].tolist(), indices, dictionaries, defaultdict(list))  
        
        highlevel_features = defaultdict(list)
        noun_df = pd.read_csv(f'results/{model_name_noslash}/{condition.lower()}_features.csv')
        for position, feature, category in zip(noun_df['Position'], noun_df['Feature'], noun_df['Category']):
            submodule_name, feature_idx = feature.split('/')
            feature_idx = int(feature_idx)
            if category in downweight_categories:
                highlevel_features[submodule_name].append((position, feature_idx, 0.))
            elif category in upweight_categories:
                highlevel_features[submodule_name].append((position, feature_idx, 2.0))
            
        intervened_gp, intervened_nongp = get_performance(model, condition_df['sentence_ambiguous'].tolist(), indices, dictionaries, highlevel_features)
        
        baseline_gp, baseline_nongp = get_performance(model, condition_df['sentence_ambiguous'].tolist(), indices, dictionaries, defaultdict(list))
# %%
