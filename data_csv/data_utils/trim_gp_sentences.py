#%%
"""
Script to check that all the garden path sentence lengths are equal
"""
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
#%%
tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
# %%
df = pd.read_csv('gp_orig.csv')
df = df[df['ambiguity'] == 'ambiguous']
# %%
original_sentences_trimmed = [' '.join(s.split()[:pos-1]) 
                              for s, pos in zip(df['Sentence'], df['disambPositionAmb'])]
GP_sentences_trimmed = [' '.join(s.split()[:pos-1]) 
                              for s, pos in zip(df['Sentence_GP'], df['disambPositionAmb'])]
correct_sentences_trimmed = [' '.join(s.split()[:pos-1]) 
                              for s, pos in zip(df['Sentence_Correct'], df['disambPositionAmb'])]
#%%
df['sentence_ambiguous'] = original_sentences_trimmed
df['sentence_gp'] = GP_sentences_trimmed
df['sentence_post'] = correct_sentences_trimmed
#%%
del df['Sentence']
del df['Sentence_GP']
del df['Sentence_Correct']
#%%
df.to_csv('gp_trimmed.csv', index=False)
# %%
