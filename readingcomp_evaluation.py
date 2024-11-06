import argparse
import csv
import torch
import random
import json
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict
from huggingface_hub import hf_hub_download
import numpy as np
import importlib
dictionary = importlib.import_module("feature-circuits-gp.dictionary_learning.dictionary")

def load_examples(datapath, tokenizer):
    examples = []
    with open(datapath, "r") as data:
        reader = csv.reader(data)
        next(reader)    # skip header
        for row in reader:
            if len(row) == 7:
                condition, is_ambig, sentence, sentence_gp, sentence_post, readingcomp_q_no, readingcomp_q_yes = row
            else:
                condition, is_ambig, sentence, readingcomp_q_no, readingcomp_q_yes = row
                
            is_ambig = (is_ambig == "True")

            sentence_tok = tokenizer(sentence, return_tensors="pt").input_ids.to("cuda")
            readingcomp_q_no_tok = tokenizer(" "+readingcomp_q_no, return_tensors="pt",
                                             add_special_tokens=False).input_ids.to("cuda")
            readingcomp_q_yes_tok = tokenizer(" "+readingcomp_q_yes, return_tensors="pt",
                                              add_special_tokens=False).input_ids.to("cuda")
            no_answer = tokenizer(" No", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            yes_answer = tokenizer(" Yes", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            if no_answer.shape[1] != 1 or yes_answer.shape[1] != 1:
                continue
            
            example_no = {
                "condition": condition,
                "ambiguous": is_ambig,
                "sentence": sentence_tok,
                "readingcomp_q": readingcomp_q_no_tok,
                "correct_answer": no_answer,
                "incorrect_answer": yes_answer
            }
            examples.append(example_no)
            example_yes = {
                "condition": condition,
                "ambiguous": is_ambig,
                "sentence": sentence_tok,
                "readingcomp_q": readingcomp_q_yes_tok,
                "correct_answer": yes_answer,
                "incorrect_answer": no_answer
            }
            examples.append(example_yes)

    return examples

def load_examples_prefix_len(dataset, num_examples, model, seed=12, pad_to_length=None, length=None,
                  ignore_patch=False):
    
    examples = []
    with open(dataset, 'r') as data:
        reader = csv.reader(data)
        next(reader)    # skip header
        for row in reader:
            condition, is_ambig, sentence, readingcomp_q_no, readingcomp_q_yes = row
            length = 8 if condition in ("NPS", "MVRR") else 9

            clean_prefix_str = f"{sentence} {readingcomp_q_yes}"
            patch_prefix_str = f"{sentence} {readingcomp_q_no}"

            clean_prefix = model.tokenizer(clean_prefix_str, return_tensors="pt",
                                            padding=False).input_ids
            patch_prefix = model.tokenizer(patch_prefix_str, return_tensors="pt",
                                            padding=False).input_ids
            clean_answer = model.tokenizer(" Yes", return_tensors="pt",
                                            padding=False).input_ids
            patch_answer = model.tokenizer(" No", return_tensors="pt",
                                            padding=False).input_ids

            clean_prefix_firstsent = sentence.split(".")[0]
            clean_prefix_firstsent_tok = model.tokenizer(clean_prefix_firstsent, return_tensors="pt",
                                                        padding=False).input_ids
            
            # remove BOS tokens from answers
            clean_answer = clean_answer[clean_answer != model.tokenizer.bos_token_id].unsqueeze(0)
            patch_answer = patch_answer[patch_answer != model.tokenizer.bos_token_id].unsqueeze(0)
            # only keep examples where answers are single tokens
            if not ignore_patch:
                if clean_prefix.shape[1] != patch_prefix.shape[1]:
                    continue
            # only keep examples where clean and patch answers are the same length
            if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
                continue
            # if we specify a `length`, filter examples if they don't match
            if length and clean_prefix_firstsent_tok.shape[1] != length:
                print(condition, clean_prefix_firstsent_tok.shape[1])
                continue
            # if we specify `pad_to_length`, left-pad all inputs to a max length
            prefix_length_wo_pad = clean_prefix.shape[1]
            if pad_to_length:
                model.tokenizer.padding_side = 'right'
                pad_length = pad_to_length - prefix_length_wo_pad
                if pad_length < 0:  # example too long
                    continue
                # left padding: reverse, right-pad, reverse
                clean_prefix = torch.flip(torch.nn.functional.pad(torch.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
                patch_prefix = torch.flip(torch.nn.functional.pad(torch.flip(patch_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
            
            example_yes = {
                "condition": condition,
                "ambiguous": True,
                "sentence": clean_prefix.to("cuda"),
                "correct_answer": clean_answer.to("cuda"),
                "incorrect_answer": patch_answer.to("cuda")
            }
            example_no = {
                "condition": condition,
                "ambiguous": True,
                "sentence": patch_prefix.to("cuda"),
                "correct_answer": patch_answer.to("cuda"),
                "incorrect_answer": clean_answer.to("cuda")
            }
            
            examples.append(example_yes)
            examples.append(example_no)
            if len(examples) >= num_examples:
                break

    return examples

def eval_example(model, prompt, correct_label, incorrect_label, ablate_features=None, inject_features=None,
                 dictionaries=None, return_surprisals=False):
    """
    model: AutoModelForCausalLM
    prompt: string
    label_pair: [token_id, token_id]
    gold_label: token_id
    ablate_features: {submodule: [feature1, feature2, ...], ...}
    """
    def _separate_positions_and_features(feature_list):
        poss = []
        feat_idxs = []
        for feature in feature_list:
            if type(feature) == tuple:
                pos = feature[0]
                feat_idx = feature[1]
            else:
                pos = None
                feat_idx = feature
            poss.append(pos)
            feat_idxs.append(feat_idx)
        if all([pos is None for pos in poss]):
            poss = None
        return poss, feat_idxs

    if ablate_features is not None or inject_features is not None:
        with model.trace(prompt), torch.no_grad():
            if ablate_features is not None:
                for submodule in ablate_features:
                    ae = dictionaries[submodule]
                    ablate_feature_list = ablate_features[submodule]
                    feature_poss, feature_idxs = _separate_positions_and_features(ablate_feature_list)

                    x = submodule.output
                    if type(x.shape) == tuple:
                        x = x[0]
                    f = ae.encode(x)
                    x_hat = ae.decode(f)
                    x_hat_orig = ae.decode(f)
                    residual = x - x_hat_orig
                    
                    f_new = torch.clone(f)
                    if feature_poss is None:
                        f_new[:, :, feature_idxs] = 0.
                    else:
                        f_new[:, feature_poss, feature_idxs] = 0.
                    x_hat = ae.decode(f_new)
                    if type(submodule.output.shape) == tuple:
                        submodule.output[0][:] = x_hat + residual
                    else:
                        submodule.output = x_hat + residual
            if inject_features is not None:
                for submodule in inject_features:
                    ae = dictionaries[submodule]
                    inject_feature_list = inject_features[submodule]
                    feature_poss, feature_idxs = _separate_positions_and_features(inject_feature_list)
                    x = submodule.output
                    if type(x.shape) == tuple:
                        x = x[0]
                    f = ae.encode(x)
                    x_hat = ae.decode(f)
                    x_hat_orig = ae.decode(f)
                    residual = x - x_hat_orig

                    f_new = torch.clone(f)
                    if feature_poss is not None:
                        f_new[:, feature_poss, feature_idxs] = 10.
                    else:
                        f_new[:, :, feature_idxs] = 10.
                    x_hat = ae.decode(f_new)
                    if type(submodule.output.shape) == tuple:
                        submodule.output[0][:] = x_hat + residual
                    else:
                        submodule.output = x_hat + residual

            # logits_saved = model.lm_head.output.save()
            logits_saved = model.embed_out.output.save()
        logits = logits_saved.value
    else:
        with model.trace(prompt), torch.no_grad():
            logits_saved = model.embed_out.output.save()
            # logits_saved = model.lm_head.output.save()
        logits = logits_saved.value
    
    logit_diff = logits[0, -1, correct_label] - logits[0, -1, incorrect_label]
    # clean_logit_diff = logits_clean[0, -1, correct_label] - logits[0, -1, incorrect_label]
    # print(logit_diff - clean_logit_diff)
    is_correct = logits[0, -1, correct_label] > logits[0, -1, incorrect_label]
    if return_surprisals:
        surprisals = -1 * torch.nn.functional.log_softmax(logits, dim=-1)
        token_strs = model.tokenizer.convert_ids_to_tokens(prompt[0])
        surprisals_tokens = [(token_strs[0], 0.0)]
        surprisals_tokens.extend([(token_strs[idx+1], logits[0, idx, prompt[0, idx+1]].item()) for idx in range(prompt.shape[-1] - 1)])
        return (is_correct, surprisals_tokens)
    return is_correct, logit_diff


def submodule_name_to_submodule(submodule_name):
    submod_type, layer_idx = submodule_name.split("_")
    layer_idx = int(layer_idx)
    if submod_type == "resid":
        # return model.model.layers[layer_idx]
        return model.gpt_neox.layers[layer_idx]
    elif submod_type == "attn":
        # return model.model.layers[layer_idx].attention
        return model.gpt_neox.layers[layer_idx].attention
    elif submod_type == "mlp":
        # return model.model.layers[layer_idx].mlp
        return model.gpt_neox.layers[layer_idx].mlp
    else:
        raise ValueError(f"Unrecognized submodule type: {submod_type}")


def load_autoencoder(autoencoder_path, submodule_name, model_type="other"):
    submod_type, layer_idx = submodule_name.split("_")
    # For Llama
    if model_type == "llama":
        ae_path = f"{autoencoder_path}/layer{layer_idx}/ae_81920.pt"
        ae = dictionary.GatedAutoEncoder(4096, 32768).to("cuda")
        ae.load_state_dict(torch.load(open(ae_path, "rb")))
    elif model_type == "gemma":
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename=f"layer_{layer_idx}/width_16k/canonical/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        ae = dictionary.JumpReLUSAE(params["W_enc"].shape[0], params["W_enc"].shape[1]).to("cuda")
        ae.load_state_dict(pt_params)
    else:
        ae_path = f"{autoencoder_path}/pythia-70m-deduped/{submod_type}_out_layer{layer_idx}/10_32768/ae.pt"
        ae = dictionary.AutoEncoder(512, 32768).to("cuda")
        ae.load_state_dict(torch.load(open(ae_path, "rb")))
    # ae = ae.half()
    return ae


def parse_features(features, dictionaries, model_type):
    feature_dict = defaultdict(list)
    for feature in features:
        if "," in feature:
            position_and_submod, feature_idx = feature.split("/")
            position, submod_name = position_and_submod.split(",")
            submod_name = submod_name.strip()
        else:
            submod_name, feature_idx = feature.split("/")
            position = None
        submodule = submodule_name_to_submodule(submod_name)
        
        if submodule not in dictionaries:
            dictionaries[submodule] = load_autoencoder(args.autoencoder_dir, submod_name,
                                                       model_type=model_type)

        if position is not None:
            feature_dict[submodule].append((int(position), int(feature_idx)))
        else:
            feature_dict[submodule].append(int(feature_idx))
    return dictionaries, feature_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="google/gemma-2-2b", help="Name of model.")
    parser.add_argument("--dataset", "-d", type=str, default="data_csv/garden_path_samelen_readingcomp.csv")
    parser.add_argument("--autoencoder_dir", "-a", type=str, default=None)
    parser.add_argument("--ablate_features", "-f", type=str, nargs='*', default=None)
    parser.add_argument("--ablate_features_file", "-af", type=str, default=None)
    parser.add_argument("--inject_features", "-i", type=str, nargs='*', default=None)
    parser.add_argument("--inject_features_file", "-if", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_bos_token=False)
    bnb_config = BitsAndBytesConfig(    # use 4-bit quantization to make it fit on a single GPU
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if "gemma-2" in args.model:
        model = LanguageModel(args.model, device_map="cuda", attn_implementation="eager")
        model_type = "gemma"
    else:
        model = LanguageModel(args.model, device_map="cuda")
        if "llama" in args.model:
            model_type = "llama"
        else:
            model_type = "other"
    
    # examples = load_examples_prefix_len(args.dataset, 200, model, ignore_patch=True)# length=)
    examples = load_examples(args.dataset, tokenizer)
    num_examples = len(examples)

    ablate_features = args.ablate_features
    if type(args.ablate_features) == str:
        ablate_features = [args.ablate_features]
    if args.ablate_features_file is not None:
        ablate_features = open(args.ablate_features_file, 'r').readlines()

    inject_features = args.inject_features
    if type(args.inject_features) == str:
        inject_features = [args.inject_features]
    if args.inject_features_file is not None:
        inject_features = open(args.inject_features_file, 'r').readlines()

    dictionaries = None
    if ablate_features is not None or inject_features is not None:
        dictionaries = {}
        if ablate_features is not None:
            dictionaries, ablate_features = parse_features(ablate_features, dictionaries,
                                                           model_type)
        if inject_features is not None:
            dictionaries, ablate_features = parse_features(inject_features, dictionaries,
                                                           model_type)

    correct = 0
    correct_grouped_amb = {"NPS": {"amb": 0, "unamb": 0}, "NPZ": {"amb": 0, "unamb": 0}, "MVRR": {"amb": 0, "unamb": 0}}
    correct_grouped_label = {"NPS": {"gp": 0, "post": 0}, "NPZ": {"gp": 0, "post": 0}, "MVRR": {"gp": 0, "post": 0}}
    total_grouped_amb = {"NPS": {"amb": 0, "unamb": 0}, "NPZ": {"amb": 0, "unamb": 0}, "MVRR": {"amb": 0, "unamb": 0}}
    total_grouped_label = {"NPS": {"gp": 0, "post": 0}, "NPZ": {"gp": 0, "post": 0}, "MVRR": {"gp": 0, "post": 0}}
    total_logit_diff = 0
    total_condition = {"NPS": 0, "NPZ": 0, "MVRR": 0}
    grouped_logit_diff = {"NPS": 0, "NPZ": 0, "MVRR": 0}
    for example in tqdm(examples, desc="Examples", total=num_examples):
        condition = example["condition"].split("_")[0]
        is_amb = "amb" if example["ambiguous"] else "unamb"
        if is_amb == "unamb":
            continue
        label = "gp" if example["correct_answer"] == tokenizer(" No", add_special_tokens=False,
                                                               return_tensors="pt").input_ids.to("cuda") else "post"
        prompt = torch.cat((example["sentence"], example["readingcomp_q"]), dim=1)
        # prompt = example["sentence"]
        is_correct, logit_diff = eval_example(model, prompt, example["correct_answer"], example["incorrect_answer"],
                                      ablate_features=ablate_features, dictionaries=dictionaries,
                                      return_surprisals=False)
        total_logit_diff += logit_diff
        is_correct = int(is_correct)
        correct += is_correct

        correct_grouped_amb[condition][is_amb] += is_correct
        total_grouped_amb[condition][is_amb] += 1
        grouped_logit_diff[condition] += logit_diff
        total_condition[condition] += 1

        if is_amb == "amb":
            correct_grouped_label[condition][label] += is_correct
            total_grouped_label[condition][label] += 1
            
    print(f"Overall logit diff: {total_logit_diff / len(examples)}")
    print(f"Overall Accuracy: {correct / num_examples:.2f}")
    for condition in correct_grouped_amb:
        print(f"{condition}:")
        for gp_post in correct_grouped_label[condition]:
            accuracy = correct_grouped_label[condition][gp_post] / total_grouped_label[condition][gp_post]
            print(f"\t{gp_post}: {accuracy}")
        print(f"\t{grouped_logit_diff[condition] / total_condition[condition]}")
