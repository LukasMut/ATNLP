# we don't want to load all libraries that are imported in this .py file into memory 
__all__ = ['load_dataset', 'sort_dict', 'w2i', 's2i']

import numpy as np
import os
import re
import torch

from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences

def load_dataset(exp:str, split:str, subdir:str='./data'):
    """load dataset into memory
    Args: 
        exp (str): experiment
        split (str): train or test dataset
    Returns:
        lang_vocab (dict): word2freq dictionary 
        w2i (dict): word2idx mapping
        i2w (dict): idx2word mapping
        lang (list): list of all sentences in either input (commands) or output (actions) language
    """
    file = subdir+exp+split+'/'+os.listdir(subdir+exp+split).pop()
    cmd_start = 'IN:'
    act_start = 'OUT:'
    cmds, acts = [], []
    cmd_vocab, act_vocab = defaultdict(int), defaultdict(int)
    
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            cmd = line[line.index(cmd_start)+len(cmd_start):line.index(act_start)].strip().split()
            act = line[line.index(act_start)+len(act_start):].strip().split()
            for w in cmd: cmd_vocab[w] += 1
            for w in act: act_vocab[w] += 1
            cmds.append(cmd)
            acts.append(act)

    cmd_vocab = sort_dict(cmd_vocab)
    act_vocab = sort_dict(act_vocab)
    # create w2i and i2w mappings
    w2i_cmds, i2w_cmds = w2i(cmd_vocab)
    w2i_acts, i2w_acts = w2i(act_vocab)
    return cmd_vocab, w2i_cmds, i2w_cmds, cmds, act_vocab, w2i_acts, i2w_acts, acts

def sort_dict(some_dict:dict): return dict(sorted(some_dict.items(), key=lambda kv:kv[1], reverse=True))

def w2i(vocab:dict):
    w2i = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    n_special_toks = len(w2i)
    for i, w in enumerate(vocab.keys()):
        w2i[w] = i + n_special_toks
    i2w = dict(enumerate(w2i.keys()))
    return w2i, i2w

def s2i(sents:list, w2i:dict, padding:bool=True, decode:bool=False):
    """sentence2idx mapping
    Args: 
        sents (list): each sentence is represented through words (str)
        w2i (dict): word2idx dictionary
        padding (str): whether padding should be performed or not (padding value = 0)
    Return:
        sents (np.ndarray): each sentence is represented through its corresponding idx in the vocab (int)
    """
    seqs = []
    n_special_toks = 2 if decode else 1
    for sent in sents:
        indices = np.zeros(len(sent) + n_special_toks, dtype=int)
        # append <SOS> token to beginning of action (!) sequences only (not to command sequences)
        if decode: 
            indices[0] = w2i['<SOS>']
        for i, w in enumerate(sent):
            indices[i + n_special_toks-1] += w2i[w]
        # append <EOS> token to end of sequence
        indices[len(indices)-1] = w2i['<EOS>']
        seqs.append(indices)
    # we have to pad sequences if we want to perform mini batch training (otherwise batch_size must be equal to 1)
    if padding:
        #NOTE: use iter() instead of list() if you don't want to load data into memory ("on-the-fly execution")
        max_len = max(iter(map(lambda seq: len(seq), seqs)))
        seqs = pad_sequences(seqs, maxlen=max_len, dtype="long", truncating="post", padding="post", value=0)
    return seqs