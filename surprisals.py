import argparse
import csv
import json
import torch
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict
from copy import deepcopy
import importlib
import numpy as np
dictionary = importlib.import_module("feature-circuits-gp.dictionary_learning.dictionary")

def load_examples(datapath, tokenizer, same_len=False):
    examples = []
    with open(datapath, "r") as data:
        reader = csv.reader(data)
        next(reader)    # skip header
        for row in reader:
            item, condition, sentence_amb, sentence_gp, sentence_post = row

            sentence_amb_tok = tokenizer(sentence_amb, return_tensors="pt").input_ids.to("cuda")
            sentence_gp_tok = tokenizer(sentence_gp, return_tensors="pt").input_ids.to("cuda")
            sentence_post_tok = tokenizer(sentence_post, return_tensors="pt").input_ids.to("cuda")

            post_continuation = tokenizer(" was", return_tensors="pt",
                                             add_special_tokens=False).input_ids.to("cuda")
            gp_tok = "." if condition in ("NPS", "MVRR") else ","
            gp_continuation = tokenizer(gp_tok, return_tensors="pt",
                                              add_special_tokens=False).input_ids.to("cuda")
            if post_continuation.shape[1] != 1 or gp_continuation.shape[1] != 1:
                continue
            
            target_len = 6 if condition in ("NPS", "MVRR") else 7
            example = {
                "condition": condition,
                "type": "amb",
                "sentence": sentence_amb_tok,
                "post_answer": post_continuation,
                "gp_answer": gp_continuation
            }
            if not same_len or example["sentence"].shape[-1] == target_len:
                examples.append(example)
            example = {
                "condition": condition,
                "type": "gp",
                "sentence": sentence_gp_tok,
                "post_answer": post_continuation,
                "gp_answer": gp_continuation
            }
            if not same_len or example["sentence"].shape[-1] == target_len:
                examples.append(example)
            example = {
                "condition": condition,
                "type": "post",
                "sentence": sentence_post_tok,
                "post_answer": post_continuation,
                "gp_answer": gp_continuation
            }
            if not same_len or example["sentence"].shape[-1] == target_len:
                examples.append(example)

    return examples
            

def eval_example(model, prompt, correct_label, incorrect_label, ablate_features=None, inject_features=None,
                 dictionaries=None, aggregation="final_pos"):
    """
    model: AutoModelForCausalLM
    prompt: string
    label_pair: [token_id, token_id]
    gold_label: token_id
    ablate_features: {submodule: [feature1, feature2, ...], ...}
    """
    if ablate_features is not None or inject_features is not None:
        with model.trace(prompt), torch.no_grad():
            if ablate_features is not None:
                for submodule in ablate_features:
                    ae = dictionaries[submodule]
                    ablate_feature_list = ablate_features[submodule]
                    x = submodule.output
                    if type(x.shape) == tuple:
                        x = x[0]
                    f = ae.encode(x)
                    x_hat = ae.decode(f)
                    x_hat_orig = ae.decode(f)
                    residual = x - x_hat_orig

                    f_new = torch.clone(f)
                    f_new[:, :, ablate_feature_list] = 0.
                    x_hat = ae.decode(f_new)
                    if type(submodule.output.shape) == tuple:
                        submodule.output[0][:] = x_hat + residual
                    else:
                        submodule.output = x_hat + residual
            if inject_features is not None:
                for submodule in inject_features:
                    ae = dictionaries[submodule]
                    inject_feature_list = inject_features[submodule]
                    x = submodule.output
                    if type(x.shape) == tuple:
                        x = x[0]
                    f = ae.encode(x)
                    x_hat = ae.decode(f)
                    x_hat_orig = ae.decode(f)
                    residual = x - x_hat_orig

                    f_new = torch.clone(f)
                    f_new[:, :5, inject_feature_list] = 5.
                    x_hat = ae.decode(f_new)
                    if type(submodule.output.shape) == tuple:
                        submodule.output[0][:] = x_hat + residual
                    else:
                        submodule.output = x_hat + residual

            logits_saved = model.lm_head.output.save()
            # logits_saved = model.embed_out.output.save()
    else:
        with model.trace(prompt), torch.no_grad():
            # logits_saved = model.embed_out.output.save()
            logits_saved = model.lm_head.output.save()

    logits = logits_saved.value
    surprisals = -1 * torch.nn.functional.log_softmax(logits, dim=-1)
    if aggregation == "final_pos":
        surprisals_gp = surprisals[0, -1, incorrect_label].item()
        surprisals_post = surprisals[0, -1, correct_label].item()
    elif aggregation == "none":
        surprisals_sent = [surprisals[0, idx, prompt[0, idx+1]].item() for idx in range(prompt.shape[1]-1)]
        surprisals_gp = surprisals_sent
        surprisals_post = deepcopy(surprisals_sent)
        surprisals_gp.append(surprisals[0, -1, incorrect_label])
        surprisals_gp = torch.Tensor(surprisals_gp)
        surprisals_post.append(surprisals[0, -1, correct_label])
        surprisals_post = torch.Tensor(surprisals_post)
    return (surprisals_post, surprisals_gp)


def submodule_name_to_submodule(submodule_name):
    submod_type, layer_idx = submodule_name.split("_")
    layer_idx = int(layer_idx)
    if submod_type == "resid":
        return model.model.layers[layer_idx]
        # return model.gpt_neox.layers[layer_idx]
    elif submod_type == "attn":
        # return model.model.layers[layer_idx].attention
        return model.gpt_neox.layers[layer_idx].attention
    elif submod_type == "mlp":
        # return model.model.layers[layer_idx].mlp
        return model.gpt_neox.layers[layer_idx].mlp
    else:
        raise ValueError(f"Unrecognized submodule type: {submod_type}")


def load_autoencoder(autoencoder_path, submodule_name, is_llama=False):
    submod_type, layer_idx = submodule_name.split("_")
    # For Llama
    if is_llama:
        ae_path = f"{autoencoder_path}/layer{layer_idx}/ae_81920.pt"
        ae = dictionary.GatedAutoEncoder(4096, 32768).to("cuda")
    else:
        ae_path = f"{autoencoder_path}/pythia-70m-deduped/{submod_type}_out_layer{layer_idx}/10_32768/ae.pt"
        ae = dictionary.AutoEncoder(512, 32768).to("cuda")
    ae.load_state_dict(torch.load(open(ae_path, "rb")))
    ae = ae.half()
    return ae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of model.")
    parser.add_argument("--dataset", "-d", type=str, default="gp_same_len.csv")
    parser.add_argument("--same_len", "-l", action="store_true")
    parser.add_argument("--autoencoder_dir", "-a", type=str, default=None)
    parser.add_argument("--ablate_features", "-f", type=str, nargs='*', default=None)
    parser.add_argument("--inject_features", "-i", type=str, nargs='*', default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_bos_token=False)
    bnb_config = BitsAndBytesConfig(    # use 4-bit quantization to make it fit on a single GPU
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
                                                # quantization_config=bnb_config)
    model = LanguageModel(args.model, torch_dtype=torch.float16,
                          device_map="cuda")
    
    examples = load_examples(args.dataset, tokenizer, same_len=args.same_len)
    num_examples = len(examples)

    ablate_features = None
    inject_features = None
    dictionaries = None
    if args.ablate_features is not None or args.inject_features:
        ablate_features = defaultdict(list)
        inject_features = defaultdict(list)
        dictionaries = {}
        if args.ablate_features is not None:
            for feature in args.ablate_features:
                submod_name, feature_idx = feature.split("/")
                submodule = submodule_name_to_submodule(submod_name)
                ablate_features[submodule].append(int(feature_idx))
                if submodule not in dictionaries:
                    dictionaries[submodule] = load_autoencoder(args.autoencoder_dir, submod_name)
        if args.inject_features is not None:
            for feature in args.inject_features:
                submod_name, feature_idx = feature.split("/")
                submodule = submodule_name_to_submodule(submod_name)
                inject_features[submodule].append(int(feature_idx))
                if submodule not in dictionaries:
                    dictionaries[submodule] = load_autoencoder(args.autoencoder_dir, submod_name)


    surprisals_gp = {"NPS": {"amb": [], "post": [], "gp": []},
                     "NPZ": {"amb": [], "post": [], "gp": []},
                     "MVRR": {"amb": [], "post": [], "gp": []}}
    surprisals_post = {"NPS": {"amb": [], "post": [], "gp": []},
                     "NPZ": {"amb": [], "post": [], "gp": []},
                     "MVRR": {"amb": [], "post": [], "gp": []}}
    totals = {"NPS": {"amb": 0, "post": 0, "gp": 0},
                "NPZ": {"amb": 0, "post": 0, "gp": 0},
                "MVRR": {"amb": 0, "post": 0, "gp": 0}}

    aggregation = "none" if args.same_len else "final_pos"
    for example in tqdm(examples, desc="Examples", total=num_examples):
        condition = example["condition"]
        type = example["type"]
        gp_continuation = example["gp_answer"]
        post_continuation = example["post_answer"]

        surprisal_post, surprisal_gp = eval_example(model, example["sentence"],
                                                    post_continuation, gp_continuation,
                                                    ablate_features=ablate_features, dictionaries=dictionaries,
                                                    aggregation=aggregation)
        
        surprisals_gp[condition][type].append(surprisal_gp)
        surprisals_post[condition][type].append(surprisal_post)
        totals[condition][type] += 1

    with open(f"analysis/word_surprisals_agg{aggregation}.jsonl", "w") as out_file: 
        for condition in surprisals_gp:
            for type in surprisals_gp[condition]:
                N = totals[condition][type]
                if args.same_len:
                    surprisals_gp_ct = torch.stack([s for s in surprisals_gp[condition][type]])
                    surprisals_post_ct = torch.stack([s for s in surprisals_post[condition][type]])
                    mean_gp_surprisal = torch.mean(surprisals_gp_ct, dim=0).tolist()
                    std_gp_surprisal = torch.std(surprisals_gp_ct, dim=0).tolist()
                    mean_post_surprisal = torch.mean(surprisals_post_ct, dim=0).tolist()
                    std_post_surprisal = torch.std(surprisals_post_ct, dim=0).tolist()
                else:
                    mean_gp_surprisal = np.mean(surprisals_gp[condition][type])
                    std_gp_surprisal = np.std(surprisals_gp[condition][type])
                    mean_post_surprisal = np.mean(surprisals_post[condition][type])
                    std_post_surprisal = np.std(surprisals_post[condition][type])
                print(f"{condition} ({type}), N={N}:")
                print(f"\tGP: {mean_gp_surprisal} ({std_gp_surprisal})")
                print(f"\tPost: {mean_post_surprisal} ({std_post_surprisal})")
                json_data = {
                    "condition": condition, "type": type, "num_examples": N,
                    "sent_gp_mean_surprisals": mean_gp_surprisal,
                    "sent_gp_std": std_gp_surprisal,
                    "sent_post_mean_surprisals": mean_post_surprisal,
                    "sent_post_std": std_post_surprisal
                }
                out_file.write(json.dumps(json_data)+"\n")
            print()