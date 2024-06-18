import argparse
import csv
import torch
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict
import importlib
dictionary = importlib.import_module("feature-circuits-gp.dictionary_learning.dictionary")

def load_examples(datapath, tokenizer):
    examples = []
    with open(datapath, "r") as data:
        reader = csv.reader(data)
        next(reader)    # skip header
        for row in reader:
            condition, is_ambig, sentence, readingcomp_q_no, readingcomp_q_yes = row
            is_ambig = (is_ambig == "True")

            sentence_tok = tokenizer(sentence, return_tensors="pt").input_ids.to("cuda")
            readingcomp_q_no_tok = tokenizer(" "+readingcomp_q_no, return_tensors="pt",
                                             add_special_tokens=True).input_ids.to("cuda")
            readingcomp_q_yes_tok = tokenizer(" "+readingcomp_q_yes, return_tensors="pt",
                                              add_special_tokens=True).input_ids.to("cuda")
            no_answer = tokenizer("No", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            yes_answer = tokenizer("Yes", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            if no_answer.shape[1] != 1 or yes_answer.shape[1] != 1:
                continue
            
            example_no = {
                "condition": condition,
                "ambiguous": is_ambig,
                "sentence": sentence_tok,
                "readingcomp_q": readingcomp_q_no_tok,
                "correct_answer": no_answer,      # we should add examples where the correct answer is yes
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
            

def eval_example(model, prompt, correct_label, incorrect_label, ablate_features=None, inject_features=None,
                 dictionaries=None):
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
                    if inject_features is not None and submodule in inject_features:
                        inject_feature_list = inject_features[submodule]
                    else:
                        inject_feature_list = []
                    x = submodule.output
                    if type(x.shape) == tuple:
                        x = x[0]
                    f = ae.encode(x)
                    x_hat = ae.decode(f)
                    x_hat_orig = ae.decode(f)
                    residual = x - x_hat_orig

                    f_new = torch.clone(f)
                    f_new[:, :, ablate_feature_list] = 0.
                    f_new[:, :5, inject_feature_list] = 5.
                    x_hat = ae.decode(f_new)
                    if type(submodule.output.shape) == tuple:
                        submodule.output[0][:] = x_hat + residual
                    else:
                        submodule.output = x_hat + residual
            if inject_features is not None:
                for submodule in inject_features:
                    if ablate_features is not None and submodule in ablate_features:
                        continue
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
        logits = logits_saved.value[0, -1]
    else:
        with model.trace(prompt), torch.no_grad():
            # logits_saved = model.embed_out.output.save()
            logits_saved = model.lm_head.output.save()
        logits = logits_saved.value[0, -1]
    return logits[correct_label] > logits[incorrect_label]


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
    parser.add_argument("--dataset", "-d", type=str, default="garden_path_readingcomp.csv")
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
    
    examples = load_examples(args.dataset, tokenizer)
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


    correct = 0
    correct_grouped_amb = {"NPS": {"amb": 0, "unamb": 0}, "NPZ": {"amb": 0, "unamb": 0}, "MVRR": {"amb": 0, "unamb": 0}}
    correct_grouped_label = {"NPS": {"gp": 0, "post": 0}, "NPZ": {"gp": 0, "post": 0}, "MVRR": {"gp": 0, "post": 0}}
    total_grouped_amb = {"NPS": {"amb": 0, "unamb": 0}, "NPZ": {"amb": 0, "unamb": 0}, "MVRR": {"amb": 0, "unamb": 0}}
    total_grouped_label = {"NPS": {"gp": 0, "post": 0}, "NPZ": {"gp": 0, "post": 0}, "MVRR": {"gp": 0, "post": 0}}
    for example in tqdm(examples, desc="Examples", total=num_examples):
        condition = example["condition"].split("_")[0]
        is_amb = "amb" if example["ambiguous"] else "unamb"
        label = "gp" if example["correct_answer"] == tokenizer("No", add_special_tokens=False,
                                                               return_tensors="pt").input_ids.to("cuda") else "post"
        prompt = torch.cat((example["sentence"], example["readingcomp_q"]), dim=1)
        is_correct = int(eval_example(model, prompt, example["correct_answer"], example["incorrect_answer"],
                                      ablate_features=ablate_features, dictionaries=dictionaries))
        correct += is_correct

        correct_grouped_amb[condition][is_amb] += is_correct
        total_grouped_amb[condition][is_amb] += 1

        if is_amb == "amb":
            correct_grouped_label[condition][label] += is_correct
            total_grouped_label[condition][label] += 1
    
    print(f"Overall Accuracy: {correct / num_examples:.2f}")
    for condition in correct_grouped_amb:
        print(f"{condition}:")
        for is_ambig in correct_grouped_amb[condition]:
            accuracy = correct_grouped_amb[condition][is_ambig] / total_grouped_amb[condition][is_ambig]
            print(f"\t{is_ambig}: {accuracy}")
        for gp_post in correct_grouped_label[condition]:
            accuracy = correct_grouped_label[condition][gp_post] / total_grouped_label[condition][gp_post]
            print(f"\t{gp_post}: {accuracy}")