import torch
import argparse
import csv
import random
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from nnsight import LanguageModel
from datasets import load_dataset


def get_mean_activations(model, num_examples=100):
    text_corpus = iter(load_dataset("monology/pile-uncopyrighted", streaming=True)["train"])
    mean_acts = {}
    for _ in range(num_examples):
        attn_acts = {}
        mlp_acts = {}
        text = next(text_corpus)["text"]
        text = model.tokenizer(text, return_tensors="pt",
                            max_length=128, padding=True, truncation=True)
        seq_len = text["input_ids"].shape[1]
        token_pos = random.randint(0, seq_len-1)
        with model.trace(text):
            for i in range(len(model.gpt_neox.layers)):
                attn_acts[f"attn_{i}"] = model.gpt_neox.layers[i].attention.output[0][0, token_pos, :].save()
                mlp_acts[f"mlp_{i}"] = model.gpt_neox.layers[i].mlp.output[0, token_pos, :].save()
                # mean_acts[f"resid_{i}"] = model.gpt_neox.layers[i].output[0][0, token_pos, :].save()
        
        for i in range(len(model.gpt_neox.layers)):
            if f"attn_{i}" not in mean_acts:
                mean_acts[f"attn_{i}"] = attn_acts[f"attn_{i}"].value
                mean_acts[f"mlp_{i}"] = mlp_acts[f"mlp_{i}"].value
            else:
                mean_acts[f"attn_{i}"] += attn_acts[f"attn_{i}"].value
                mean_acts[f"mlp_{i}"] += mlp_acts[f"mlp_{i}"].value

    for i in range(len(model.gpt_neox.layers)):
        mean_acts[f"attn_{i}"] /= num_examples
        mean_acts[f"mlp_{i}"] /= num_examples

    return mean_acts


def estimate_effects(model, mean_activations, input, correct_token, incorrect_token):
    clean_act_cache = {}
    patch_act_cache = {}
    effects = {}

    # clean run
    with model.trace(input):
        for layer in range(model.config.num_hidden_layers):
            mlp_name = f"mlp_{layer}"
            attn_name = f"attn_{layer}"
            model.gpt_neox.layers[layer].mlp.output.retain_grad()
            model.gpt_neox.layers[layer].attention.output[0].retain_grad()
            clean_act_cache[mlp_name] = model.gpt_neox.layers[layer].mlp.output.save()
            clean_act_cache[attn_name] = model.gpt_neox.layers[layer].attention.output.save()
        
        logits = model.embed_out.output.save()
    metric = logits.value[:, -1, incorrect_token] - \
             logits.value[:, -1, correct_token]
    metric.sum().backward(retain_graph=True)

    # patch run
    # with model.trace(input):
    #     for layer in range(model.config.num_hidden_layers):
    #         mlp_name = f"mlp_{layer}"
    #         attn_name = f"attn_{layer}"
    #         patch_act_cache[mlp_name] = model.gpt_neox.layers[layer].mlp.output.save()
    #         patch_act_cache[attn_name] = model.gpt_neox.layers[layer].attention.output.save()

    for submodule in clean_act_cache:
        if isinstance(clean_act_cache[submodule].value, tuple):
            clean_act = clean_act_cache[submodule].value[0]
            # patch_act = patch_act_cache[submodule].value[0]
        else:
            clean_act = clean_act_cache[submodule].value
            # patch_act = patch_act_cache[submodule].value
        effects[submodule] = clean_act.grad * (mean_activations[submodule] - clean_act).detach()
    
    return effects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m-deduped")
    parser.add_argument("--dataset_path", type=str, default="gp_orig.csv")
    args = parser.parse_args()

    model = LanguageModel(args.model, dispatch=True, device_map="cuda:0")
    mean_effects = defaultdict(dict)
    num_examples = len(open(args.dataset_path, "r").readlines()) - 1
    num_examples_condition = defaultdict(int)

    mean_activations = get_mean_activations(model)

    with open(args.dataset_path, 'r') as gp_sentences:
        idx = 0
        reader = csv.reader(gp_sentences)
        next(reader)
        for row in tqdm(reader, desc="Examples", total=num_examples):
            _, condition, disamb_pos_amb, disamb_pos_unamb, ambiguity, sentence, _, _, _ = row
            disamb_pos = disamb_pos_amb if ambiguity == "ambiguous" else disamb_pos_unamb
            disamb_pos = int(disamb_pos) - 1
            condition = condition.split("_")[0] + f"_{ambiguity}"

            disamb_word = " " + sentence.split()[disamb_pos]
            disamb_word_tok = model.tokenizer(disamb_word, return_tensors="pt").input_ids
            if disamb_word_tok.shape[-1] != 1:
                continue
            num_examples_condition[condition] += 1
            
            garden_path_prefix = " ".join(sentence.split()[:disamb_pos])
            alternate_completion = "." if (condition.startswith("NPS") or condition.startswith("MVRR")) else ","
            alternate_completion_tok = model.tokenizer(alternate_completion, return_tensors="pt").input_ids

            submodule_effects = estimate_effects(model, mean_activations,
                                                 garden_path_prefix, disamb_word_tok, alternate_completion_tok)
            for submodule in submodule_effects:
                submodule_effects[submodule] = submodule_effects[submodule].sum(dim=1)
                if condition not in mean_effects[submodule]:
                    mean_effects[submodule][condition] = submodule_effects[submodule]
                else:
                    mean_effects[submodule][condition] += submodule_effects[submodule]
            idx += 1
            if idx > 3:
                break
    
    mlp_effects = defaultdict(lambda: torch.zeros((model.config.num_hidden_layers, model.config.hidden_size)).to("cuda"))
    attn_effects = defaultdict(lambda: torch.zeros((model.config.num_hidden_layers, model.config.hidden_size)).to("cuda"))
    for submodule in mean_effects:
        for condition in mean_effects[submodule]:
            mean_effects[submodule][condition] /= num_examples_condition[condition]
            print(mean_effects[submodule][condition].shape)
            layer = int(submodule.split("_")[1])
            if "mlp" in submodule:
                if layer in mlp_effects[condition]:
                    mlp_effects[condition][layer] += mean_effects[submodule][condition][0]
                else:
                    mlp_effects[condition][layer] = mean_effects[submodule][condition][0]
            else:
                if layer in attn_effects[condition]:
                    attn_effects[condition][layer] += mean_effects[submodule][condition][0]
                else:
                    attn_effects[condition][layer] = mean_effects[submodule][condition][0]
    
    top_mlps, bottom_mlps, top_heads, bottom_heads = {}, {}, {}, {}
    for condition in mlp_effects:
        top_mlps[condition] = torch.topk(mlp_effects[condition].cpu().flatten(), 20)
        top_mlps[condition] = (top_mlps[condition][0], np.array(np.unravel_index(top_mlps[condition][1].numpy(), mlp_effects[condition].shape)).T)
        bottom_mlps[condition] = torch.topk(mlp_effects[condition].cpu().flatten(), 20, largest=False)
        bottom_mlps[condition] = (bottom_mlps[condition][0], np.array(np.unravel_index(bottom_mlps[condition][1].numpy(), mlp_effects[condition].shape)).T)

        top_heads[condition] = torch.topk(attn_effects[condition].cpu().flatten(), 20)
        top_heads[condition] = (top_heads[condition][0], np.array(np.unravel_index(top_heads[condition][1].numpy(), attn_effects[condition].shape)).T)
        bottom_heads[condition] = torch.topk(attn_effects[condition].cpu().flatten(), 20, largest=False)
        bottom_heads[condition] = (bottom_heads[condition][0], np.array(np.unravel_index(bottom_heads[condition][1].numpy(), attn_effects[condition].shape)).T)
    
    # similarities = []
    # for condition1 in mlp_effects:
    #     similarities.append([])
    #     for condition2 in mlp_effects:
    #         # Jaccard similarity: intersection / union
    #         top_mlp_set1 = set([tuple(e) for e in top_mlps[condition1]])
    #         top_mlp_set2 = set([tuple(e) for e in top_mlps[condition2]])
    #         bottom_mlp_set1 = set([tuple(e) for e in bottom_mlps[condition1]])
    #         bottom_mlp_set2 = set([tuple(e) for e in bottom_mlps[condition2]])
    #         tops_jaccard = len(top_mlp_set1.intersection(top_mlp_set2)) / len(top_mlp_set1.union(top_mlp_set2))
    #         similarities[-1].append(tops_jaccard)
    # for condition1 in mlp_effects:
    #     similarities.append([])
    #     for condition2 in mlp_effects:
    #         # Jaccard similarity: intersection / union
    #         top_mlp_set1 = set([tuple(e) for e in top_mlps[condition1]])
    #         top_mlp_set2 = set([tuple(e) for e in top_mlps[condition2]])
    #         bottom_mlp_set1 = set([tuple(e) for e in bottom_mlps[condition1]])
    #         bottom_mlp_set2 = set([tuple(e) for e in bottom_mlps[condition2]])
    #         tops_jaccard = len(top_mlp_set1.intersection(top_mlp_set2)) / len(top_mlp_set1.union(top_mlp_set2))
    #         similarities[-1].append(tops_jaccard)

    for condition in top_mlps:
        print("---")
        print(condition)
        print("\tTop neurons:", top_mlps[condition][1])
        print("\t", top_mlps[condition][0])
        print()
        print("\tTop heads:", top_heads[condition][1])
        print("\t", top_heads[condition][0])
        # print()
        # print("---")
        # print()
        # print("\tBottom neurons:", np.array(np.unravel_index(bottom_mlp[1].numpy(), mlp_effects.shape)).T)
        # print(bottom_mlp[0])
        # print()
        # print("\tBottom heads:", np.array(np.unravel_index(bottom_attn[1].numpy(), attn_effects.shape)).T)
        # print(bottom_attn[0])