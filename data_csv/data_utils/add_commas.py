#%%
import pandas as pd
#%%
df = pd.read_csv('data_csv/gp_same_len.csv')

#%%
npz_df = df[df['condition'] == 'NPZ']
ambiguous_npz_comma, gp_npz_comma, post_npz_comma = [], [], []
for ambiguous, gp, post in zip(npz_df['sentence_ambiguous'], npz_df['sentence_gp'], npz_df['sentence_post']):
    ambiguous = ambiguous.split()
    ambiguous.insert(4, ',')
    ambiguous = ' '.join(ambiguous).replace(' ,', ',')
    post = post.split()
    post.insert(4, ',')
    post = ' '.join(post).replace(' ,', ',')
    
    ambiguous_npz_comma.append(ambiguous)
    gp_npz_comma.append(gp)
    post_npz_comma.append(post)
# %%
d = df.to_dict(orient='list')
#%%
d['sentence_ambiguous'] += ambiguous_npz_comma
d['sentence_gp'] += gp_npz_comma
d['sentence_post'] += post_npz_comma
d['condition'] += ['NPZ_comma']*len(ambiguous_npz_comma)
d['item'] += range(1, len(ambiguous_npz_comma)+1)

# %%
df2 = pd.DataFrame(d)
# %%
df2.to_csv('data_csv/gp_same_len_with_comma.csv', index=False)
