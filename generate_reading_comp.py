import csv
import sys
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import defaultdict

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = "garden_path_readingcomp.csv"
    reader = csv.reader(open(in_file, 'r'))
    lemmatizer = WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()
    
    with open(out_file, 'w') as out_data:
        writer = csv.writer(out_data)
        writer.writerow(["condition", "ambiguous", "Sentence", "Comp_Question"])   # write header
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
                new_words = ["Did"]
                for idx, (word, pos_tag) in enumerate(tagged_words):  # lemmatize first verb in sentence
                    if word == "who" or word == "that":
                        continue
                    if word == "was" or word == "were":
                        continue
                    if pos_tag.startswith("V"):
                        verb_inf = lemmatizer.lemmatize(word, pos="v")
                        words[idx] = verb_inf
                        new_words.append(words[idx])
                        continue
                    if idx == 0:
                        word = word.lower()
                    new_words.append(word)
                    if idx == disamb_pos - 1:
                        break
                new_words.append("?")

            elif condition.startswith("NPZ"):
                new_words = ["Did"]
                for idx, (word, pos_tag) in enumerate(tagged_words):
                    if idx == 0:
                        continue
                    if word == ",":
                        disamb_pos += 1
                        continue
                    if pos_tag.startswith("V"):
                        verb_inf = lemmatizer.lemmatize(word, pos="v")
                        words[idx] = verb_inf
                        new_words.append(words[idx])
                        continue
                    if idx == 1:
                        word = word.lower()
                    new_words.append(word)
                    if idx == disamb_pos - 1:
                        break
                new_words.append("?")

            else:
                raise ValueError(f"Unrecognized condition: {condition}")

            readingcomp_q = detokenizer.detokenize(new_words)
            writer.writerow([condition, is_amb, sentence, readingcomp_q])
