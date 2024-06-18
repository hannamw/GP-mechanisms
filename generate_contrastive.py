import csv
import json
import sys
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import defaultdict

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = "garden_path_constrastive.json"
    out_prefix = "garden_path_"
    reader = csv.reader(open(in_file, 'r'))
    lemmatizer = WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()
    
    with open(out_file, 'w') as out_data:
        next(reader)            # skip header
        for row in reader:
            _, condition, disamb_pos_amb, disamb_pos_unamb, amb, sentence, _, _, _ = row
            disamb_pos_amb = int(disamb_pos_amb) - 1       # convert to 0-index
            disamb_pos_unamb = int(disamb_pos_unamb) - 1   # convert to 0-index
            words = nltk.tokenize.word_tokenize(sentence)
            tagged_words = nltk.pos_tag(words)
            is_amb = (amb == "ambiguous")
            disamb_pos = disamb_pos_amb if is_amb else disamb_pos_unamb

            if condition.startswith("NPS") or condition.startswith("MVRR"):
                clean_prefix = " ".join(words[:disamb_pos])
                clean_answer = " "+words[disamb_pos]
                patch_answer = "."

            elif condition.startswith("NPZ"):
                if amb == "unambiguous":
                    disamb_pos += 1
                clean_prefix = " ".join(words[:disamb_pos]).replace(" ,", ",")
                clean_answer = " "+words[disamb_pos]
                patch_answer = ","
            
            example = {
                "clean_prefix": clean_prefix,
                "clean_answer": clean_answer,
                "patch_answer": patch_answer,
                "patch_prefix": ""
            }
            out_data.write(json.dumps(example) + "\n")

            # write to condition-specific files
            condition_name = condition.split("_")[0] + "_" + amb
            with open(out_prefix+condition_name+".json", "a") as condition_out:
                condition_out.write(json.dumps(example) + "\n")

            print(clean_prefix)
            print(clean_answer, " ||| ", patch_answer)
            print()