# we don't want to load all libraries that are imported in this .py file into memory 
__all__ = ['load_dataset', 'sort_dict', 'w2i', 'create_pairs', 's2i', 'pairs2idx']

import numpy as np
import os
import re
import torch

from collections import defaultdict
from torch.autograd import Variable
#TODO: import TensorDataset to load source-target language pairs more efficiently
from torch.utils.data import TensorDataset

def load_dataset(exp:str, split:str, subdir:str='./data'):
    """load dataset into memory
    Args: 
        exp (str): experiment (one of [exp_1a, exp_1b, exp_2, exp_3])
        split (str): train or test dataset
    Returns:
        lang_vocab (dict): word2freq dictionary 
        w2i (dict): word2idx mapping
        i2w (dict): idx2word mapping
        lang (list): list of all sentences in either input (commands) or output (actions) language
    """
    assert isinstance(exp, str), 'experiment must be one of [exp_1a, exp_1b, exp_2, exp_3]'
    file = subdir+exp+split+'/'+os.listdir(subdir+exp+split).pop()
    cmd_start = 'IN:'
    act_start = 'OUT:'
    cmds, acts = [], []
    cmd_vocab, act_vocab = defaultdict(int), defaultdict(int)
    
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            #TODO: figure out whether "I_" at the beginning of each action has to be stripped (seems odd)
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
    w2i = {'<SOS>': 0, '<EOS>': 1, '<UNK>': 2}
    n_special_toks = len(w2i)
    for i, w in enumerate(vocab.keys()):
        w2i[w] = i + n_special_toks
    i2w = dict(enumerate(w2i.keys()))
    return w2i, i2w

def create_pairs(cmds:list, acts:list): return list(zip(cmds, acts))

def s2i(sent:list, w2i:dict, decode:bool=False):
    indices = [w2i['<SOS>']] if decode else []
    for w in sent:
        indices.append(w2i[w])
    indices.append(w2i['<EOS>'])
    return indices

def pairs2idx(cmd_act_pair:tuple, w2i_cmd:dict, w2i_act:dict):
    """command-action pair to indices mapping
    Args:
        cmd_act_pair (list): each action / command is represented through strings
        w2i_cmd (dict): word2idx input language dictionary (commands)
        w2i_act (dict): word2idx output language dictionary (actions) 
    Return:
        command-action pair (tuple): each sentence is represented through a torch.tensor of corresponding indices in the vocab
    """
    cmd, act = cmd_act_pair
    cmd_seq = Variable(torch.tensor(s2i(cmd, w2i_cmd), dtype=torch.long))
    act_seq = Variable(torch.tensor(s2i(act, w2i_act, decode=True), dtype=torch.long))
    return (cmd_seq, act_seq)


#TODO: we might want to implement a shuffled tensor dataloader that does not exploit random.choice