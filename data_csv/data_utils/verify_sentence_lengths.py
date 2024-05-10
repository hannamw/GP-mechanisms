#%%
"""
Script to check that all the garden path sentence lengths are equal
"""
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
#%%
#tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
tok = AutoTokenizer.from_pretrained('gpt2')
# %%
df = pd.read_csv('data_csv/gp_same_len.csv')
if 'ambiguity' in df.columns:
    df = df[df['ambiguity'] == 'ambiguous']
df = df[df['condition'] != 'NPZ_comma']
# %%
original_sentence_lens = np.array([len(tok(s).input_ids) for s in df['sentence_ambiguous']])
GP_sentence_lens = np.array([len(tok(s).input_ids) for s in df['sentence_gp']])
correct_sentence_lens = np.array([len(tok(s).input_ids) for s in df['sentence_post']])
# %%
assert np.all(original_sentence_lens == GP_sentence_lens) and \
np.all(original_sentence_lens == correct_sentence_lens)
# %%
