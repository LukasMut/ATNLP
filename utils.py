__all__ = ['load_dataset', 'sort_dict', 'w2i', 'create_pairs', 'max_length', 's2i', 'pairs2idx', 'create_batches']

import numpy as np
import os
import re
import torch

from collections import defaultdict, Counter
from keras.preprocessing.sequence import pad_sequences
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

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
    if exp=='/exp_3': assert len(subexp) > 0, 'subexp must be one of {primitive, primitive_extended}'
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
    w2i = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    n_special_toks = len(w2i)
    for i, w in enumerate(vocab.keys()):
        w2i[w] = i + n_special_toks
    i2w = dict(enumerate(w2i.keys()))
    return w2i, i2w

def create_pairs(cmds:list, acts:list): return list(zip(cmds, acts))

def s2i(sent:list, w2i:dict, decode:bool):
    indices = [w2i['<SOS>']] if decode else []
    for w in sent:
        indices.append(w2i[w])
    indices.append(w2i['<EOS>'])
    return np.array(indices)

def max_length(seqs:list): return max((len(seq) for seq in seqs))

def zero_padding(seqs:list, max_length:int, pad_token:int):
    return pad_sequences(seqs, maxlen=max_length, dtype='int64', padding='post', truncating='post', value=pad_token)

def pairs2idx(cmds:list, acts:list, w2i_cmd:dict, w2i_act:dict, padding:bool=True, training:bool=True, reverse_source:bool=False):
    """command-action pair to indices mapping
    Args:
        commands (list): natural language commands (each command is represented through strings)
        actions (list): target actions (each action is represented through strings)
        w2i_cmd (dict): word2idx input language dictionary (commands)
        w2i_act (dict): word2idx output language dictionary (actions)
        padding (bool): specifies whether zero-padding should be performed
    Return:
        padded command-action pairs (tuple): each sentence is represented through a torch.tensor of corresponding indices in the vocab
    """
    pad_token = 0
    if padding:
        cmd_sequences = [s2i(cmd, w2i_cmd, decode=False) for cmd in cmds]
        act_sequences = [s2i(act, w2i_act, decode=True) for act in acts]
        maxlen_cmds = max_length(cmd_sequences)
        maxlen_acts = max_length(act_sequences)
        cmd_sequences = zero_padding(cmd_sequences, maxlen_cmds, pad_token)
        act_sequences = zero_padding(act_sequences, maxlen_acts, pad_token)
        
        if training:
            act_masks = torch.zeros((act_sequences.shape[0], maxlen_acts), dtype=torch.bool).to(device)
            for i, act in enumerate(act_sequences):
                act_masks[i, act != pad_token] = 1
    else:
        cmd_sequences = np.array([s2i(cmd, w2i_cmd, decode=False) for cmd in cmds])
        act_sequences = np.array([s2i(act, w2i_act, decode=True) for act in acts])
        
    if reverse_source:
        # reverse the order of words in the source sentence
        cmd_sequences = cmd_sequences[::-1]
        
    cmd_sequences = Variable(torch.tensor(cmd_sequences, dtype=torch.long).to(device))
    act_sequences = Variable(torch.tensor(act_sequences, dtype=torch.long).to(device))
    input_lengths = torch.tensor([len(seq[seq != pad_token]) for seq in cmd_sequences], dtype=torch.long).to(device)
    
    if training:
        return cmd_sequences, act_sequences, input_lengths, act_masks
    else:
        return cmd_sequences, act_sequences, input_lengths

def create_batches(cmds:torch.Tensor, acts:torch.Tensor, input_lengths:torch.Tensor, batch_size:int, masks:bool=None,
                   split:str='train', num_samples:bool=None):
    """creates mini-batches of source-target pairs
    Args:
        cmds (torch.tensor): command sequences
        acts (torch.tensor): action sequences
        input_lenghts (torch.tensor): number of non-<PAD> tokens per sequence
        batch_size (int): number of sequences in each mini-batch
        masks (torch.tensor): masks for loss have to passed during training but not during inference time
        split (str): training or testing
        num_samples (int): number of samples to draw while creating mini-batches (equivalent to number of iterations)
    Return:
        PyTorch data loader (DataLoader object)
    """
    data = TensorDataset(cmds, input_lengths, acts, masks) if isinstance(masks, torch.Tensor) else TensorDataset(cmds, input_lengths, acts)
    if split == 'train':
        isinstance(num_samples, int), 'number of samples to draw must be specified if split is training'
        # during training randomly sample elements from the train set 
        sampler = RandomSampler(data, replacement=True, num_samples=num_samples)
    elif split == 'test':
        # during testing sequentially sample elements from the test set (i.e., always sample in the same order)
        sampler = SequentialSampler(data)
    # sampler and shuffle are mutually exclusive (no shuffling for testing, random sampling for training)
    dl = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler)
    return dl