from functools import partial
from pathlib import Path

import pandas as pd 

import torch 
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer

cols = {'sentence_ambiguous' , 'sentence_post', 'sentence_gp'}

def collate_fn(xs):
    clean, corrupted, label = zip(*xs)
    return list(clean), list(corrupted), torch.tensor(label) 

class GPDataset(Dataset):
    def __init__(self, model, df, clean='sentence_ambiguous', corrupted='sentence_post', token_id_labels=True):
        assert clean in cols and corrupted in cols, f'one of clean:"{clean}" or corrupted:"{corrupted}" was not in {cols}'
        assert clean != corrupted
        self.df = df 
        self.clean=clean 
        self.corrupted=corrupted

        self.token_id_labels = token_id_labels

        self.period_token = model.tokenizer('.', add_special_tokens=False).input_ids[0] if self.token_id_labels else '.'
        self.comma_token = model.tokenizer(',', add_special_tokens=False).input_ids[0] if self.token_id_labels else ','
        self.was_token = model.tokenizer(' was', add_special_tokens=False).input_ids[0] if self.token_id_labels else ' was'

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row =  self.df.loc[index]
        gp_label = self.comma_token if row['condition'] == 'NPZ_UAMB' else self.period_token
        if self.clean == 'sentence_gp':
            label = [gp_label, self.was_token]
        elif self.clean == 'sentence_post':
            label = [self.was_token, gp_label]
        else:
            if self.corrupted == 'sentence_gp':
                label = [self.was_token, gp_label]
            else:
                label = [gp_label, self.was_token]

        return row[self.clean], row[self.corrupted], label
    
    def name(self):
        return f'{self.clean[9:]}-{self.corrupted[9:]}'

    def to_dataloader(self, batch_size:int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

def topk(model, s, k=5):
    with torch.inference_mode():
        logits = model(s).squeeze(0)[-1]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k=k)
    for idx, val in zip(top.indices, top.values):
        print(f"{model.tokenizer.decode(idx)}:\t{val:.3f}")

def get_prob(model, s, t):
    idx = model.tokenizer(t, add_special_tokens=False).input_ids[0]
    with torch.inference_mode():
        logits = model(s).squeeze(0)[-1]
    probs = torch.softmax(logits, dim=-1)
    return probs[idx]
#%%
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def kl_div(logits, clean_logits, input_lengths, labels, loss=True, mean=True):
    logits = get_logit_positions(logits, input_lengths)
    clean_logits = get_logit_positions(clean_logits, input_lengths)

    log_probs = torch.log_softmax(logits, dim=-1)
    clean_log_probs = torch.log_softmax(clean_logits, dim=-1)

    results = torch.nn.functional.kl_div(log_probs, clean_log_probs, log_target=True, reduction='none').mean(-1)
    return results.mean() if mean else results

def logit_diff(logits, clean_logits, input_lengths, labels, loss=True, mean=True):
    logits = get_logit_positions(logits, input_lengths)
    labels = labels.to(logits.device)

    correct_label = logits[torch.arange(logits.size(0), device=logits.device), labels[:, 0]]
    incorrect_label = logits[torch.arange(logits.size(0), device=logits.device), labels[:, 1]]

    logit_diff = correct_label - incorrect_label

    if loss:
        logit_diff *= -1

    if mean:
        logit_diff = logit_diff.mean()
    
    return logit_diff