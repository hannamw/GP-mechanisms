import csv
import json
import os
from collections import defaultdict

if __name__ == "__main__":
    with open("../garden_path_readingcomp.csv", 'r') as gp_comp:
        reader = csv.reader(gp_comp)
        comp_examples = defaultdict(list)
        for row in reader:
            condition, ambiguous, sentence, comp_question_gp, comp_question_post = row
            gp_comp_prompt = f"{sentence} {comp_question_gp}"
            post_comp_prompt = f"{sentence} {comp_question_post}"
            json_data_gp = {
                "clean_prefix": gp_comp_prompt,
                "patch_prefix": "",
                "clean_answer": " No",
                "patch_answer": " Yes",
            }
            json_data_post = {
                "clean_prefix": post_comp_prompt,
                "patch_prefix": "",
                "clean_answer": " Yes",
                "patch_answer": " No"
            }

            condition_name = condition.split("_")[0]
            if ambiguous == "True":
                filename_gp = f"{condition_name}_ambiguous_gp_readingcomp.json"
                filename_post = f"{condition_name}_ambiguous_post_readingcomp.json"
            else:
                filename_gp = f"{condition_name}_unambiguous_gp_readingcomp.json"
                filename_post = f"{condition_name}_unambiguous_post_readingcomp.json"
            comp_examples[filename_gp].append(json_data_gp)
            comp_examples[filename_post].append(json_data_post)
    
    for filename in comp_examples:
        out_path = os.path.join("../../feature-circuits-gp/data/", filename)
        with open(out_path, 'w') as out_data:
            for example in comp_examples[filename]:
                out_data.write(json.dumps(example)+"\n")