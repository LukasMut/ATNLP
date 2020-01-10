# we don't want to load all libraries that are imported in this .py file into memory 
__all__ = ['load_dataset', 'semantic_mapping', 'sort_dict', 'w2i', 'create_pairs', 's2i', 'pairs2idx']

import numpy as np
import os
import re
import torch

from collections import defaultdict
from torch.autograd import Variable
#TODO: import TensorDataset to load source-target language pairs more efficiently
from torch.utils.data import TensorDataset

def semantic_mapping(command:list):
    """Semantic mapping of command to action sequences according to phrase-structure grammar specified in SCAN task
    Arg:
        command (list) - list of command strings until first occurrence of conjunction
    Return:    
        action (list) - command strings translated into action strings
    """
    u = {'walk', 'look', 'run', 'jump'}
    if 'left' in command:
        direction = 'I_TURN_LEFT' 
    elif 'right' in command:
        direction = 'I_TURN_RIGHT'
    else:
        direction = ''
    prim = True
    for w in u:
        if w in command:
            prim = False
            w = 'I_' + w.upper() # prepend upper-case I and underscore to action word
            break
    if 'opposite' in command:
        if prim:
            action = direction + ' ' + direction + ' '
        else:
            action = direction + ' ' + direction + ' ' + w + ' '
    else:
        n = 4 if 'around' in command else 1
        if prim:
            action = (direction + ' ') * n  
        else:
            action = (direction + ' ' + w + ' ') * n
    if 'thrice' in command:
        rep = 3
    elif 'twice' in command:
        rep = 2
    else:
        rep = 1
    return (action * rep).strip().split()

def load_dataset(exp:str, split:str, subdir:str='./data', subexp:str='', remove_conj:bool=False, sequence_copying:bool=False):
    """load dataset into memory
    Args: 
        exp (str): experiment (one of [exp_1a, exp_1b, exp_2, exp_3])
        split (str): train or test dataset
        subexp (str): subexperiment with different primitives (only necessary for experiment 3)
        remove_conj (bool): specifies whether conjunctions (i.e., 'and', 'after') should be removed from commands
        sequence_copying (bool): specifies whether copy of word sequence x_i should be appended to original word sequence x_i
    Returns:
        lang_vocab (dict): word2freq dictionary
        w2i (dict): word2idx mapping
        i2w (dict): idx2word mapping
        lang (list): list of all sentences in either input (commands) or output (actions) language
    """
    assert isinstance(exp, str), 'experiment must be one of {/exp_1, /exp_1, /exp_2, /exp_3}'
    if exp=='/exp_3': assert len(subexp) > 0, 'subexp must be one of {turn_left, jump}'
    file = subdir+exp+subexp+split+'/'+os.listdir(subdir+exp+subexp+split).pop()
    cmd_start = 'IN:'
    act_start = 'OUT:'
    cmds, acts = [], []
    cmd_vocab, act_vocab = defaultdict(int), defaultdict(int)

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            cmd = line[line.index(cmd_start)+len(cmd_start):line.index(act_start)].strip().split()
            act = line[line.index(act_start)+len(act_start):].strip().split()
            if remove_conj:
                if re.search('after', ' '.join(cmd)):
                    conj_idx = cmd.index('after')
                    #NOTE: if remove conj, reverse the order of action sequence ("X_1 after X_2 -> X_2 X_1" -> "X_1 X_2 -> X_1 X_2")
                    x_1 = semantic_mapping(cmd[:conj_idx])
                    x_2 = act[:len(act)-len(x_1)]
                    x_1.extend(x_2)
                    act = x_1
                    cmd.pop(conj_idx)
                elif re.search('and', ' '.join(cmd)):
                    conj_idx = cmd.index('and')
                    cmd.pop(conj_idx)
            if sequence_copying:
                act.extend(act)
                cmd.extend(cmd)
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
    # NOTE: with batch_size = 1 we don't make use of the special <PAD> token (only necessary for mini-batch training)
    w2i = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
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


#TODO: we might want to implement a shuffled tensor dataloader that does not exploit random.choice (crucial for mini-batch training)