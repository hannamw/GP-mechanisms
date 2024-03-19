import argparse
import csv
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_examples(datapath, tokenizer):
    examples = []
    with open(datapath, "r") as data:
        reader = csv.reader(data)
        next(reader)    # skip header
        for row in reader:
            condition, is_ambig, sentence, readingcomp_q_no, readingcomp_q_yes = row
            is_ambig = (is_ambig == "True")

            sentence_tok = tokenizer(sentence, return_tensors="pt").input_ids.to("cuda")
            readingcomp_q_no_tok = tokenizer(" "+readingcomp_q_no, return_tensors="pt").input_ids.to("cuda")
            readingcomp_q_yes_tok = tokenizer(" "+readingcomp_q_yes, return_tensors="pt").input_ids.to("cuda")
            no_answer = tokenizer("No", return_tensors="pt").input_ids.to("cuda")
            yes_answer = tokenizer("Yes", return_tensors="pt").input_ids.to("cuda")
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
            

def eval_example(model, prompt, correct_label, incorrect_label):
    """
    model: AutoModelForCausalLM
    prompt: string
    label_pair: [token_id, token_id]
    gold_label: token_id
    """
    logits = model(prompt).logits[0, -1]
    return logits[correct_label] > logits[incorrect_label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf", help="Name of model.")
    parser.add_argument("--dataset", "-d", type=str, default="garden_path_readingcomp.csv")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_bos_token=False)
    # bnb_config = BitsAndBytesConfig(    # use 4-bit quantization to make it fit on a single GPU
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda")
                                                #  quantization_config=bnb_config)
    
    examples = load_examples(args.dataset, tokenizer)
    num_examples = len(examples)

    correct = 0
    correct_grouped = {"NPS": {"amb": 0, "unamb": 0}, "NPZ": {"amb": 0, "unamb": 0}, "MVRR": {"amb": 0, "unamb": 0}}
    total_grouped = {"NPS": {"amb": 0, "unamb": 0}, "NPZ": {"amb": 0, "unamb": 0}, "MVRR": {"amb": 0, "unamb": 0}}
    for example in tqdm(examples, desc="Examples", total=num_examples):
        condition = example["condition"].split("_")[0]
        is_amb = "amb" if example["ambiguous"] else "unamb"
        prompt = torch.cat((example["sentence"], example["readingcomp_q"]), dim=1)
        is_correct = int(eval_example(model, prompt, example["correct_answer"], example["incorrect_answer"]))
        correct += is_correct

        correct_grouped[condition][is_amb] += is_correct
        total_grouped[condition][is_amb] += 1
    
    print(f"Overall Accuracy: {correct / num_examples:.2f}")
    for condition in correct_grouped:
        print(f"{condition}:")
        for is_ambig in correct_grouped[condition]:
            accuracy = correct_grouped[condition][is_ambig] / total_grouped[condition][is_ambig]
            print(f"\t{is_ambig}: {accuracy}")