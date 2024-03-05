#%%
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
#%%
tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
# %%
df = pd.read_csv('garden_path_sentences.csv')
df = df[df['ambiguity'] == 'ambiguous']
# %%
original_sentences_trimmed = [' '.join(s.split()[:pos-1]) 
                              for s, pos in zip(df['Sentence'], df['disambPositionAmb'])]
GP_sentences_trimmed = [' '.join(s.split()[:pos-1]) 
                              for s, pos in zip(df['Sentence_GP'], df['disambPositionAmb'])]
correct_sentences_trimmed = [' '.join(s.split()[:pos-1]) 
                              for s, pos in zip(df['Sentence_Correct'], df['disambPositionAmb'])]
# %%
original_sentence_lens = np.array([len(tok(s).input_ids) for s in original_sentences_trimmed])
GP_sentence_lens = np.array([len(tok(s).input_ids) for s in GP_sentences_trimmed])
correct_sentence_lens = np.array([len(tok(s).input_ids) for s in correct_sentences_trimmed])
# %%
assert np.all(original_sentence_lens == GP_sentence_lens) and \
np.all(original_sentence_lens == correct_sentence_lens)
# %%
