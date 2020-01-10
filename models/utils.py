__all__ = ['train', 'test', 'cosine_similarity', 'compute_similarities', 'sample_distinct_pairs']

import numpy as np
import torch.nn as nn
import random
import re 
import torch

from collections import defaultdict
from itertools import islice
from sklearn.utils import shuffle
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import TensorDataset

from models.Encoder import *
from models.Decoder import *
from utils import pairs2idx, s2i, semantic_mapping

# set fixed random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = ("cuda" if torch.cuda.is_available() else "cpu")
    
### Training ###

def train(train_dl, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, epochs:int, batch_size:int,
          learning_rate:float=1e-3, max_ratio:float=0.95, min_ratio:float=0.15, detailed_analysis:bool=True,
          detailed_results:bool=False, similarity_computation:bool=False):
    
    # <PAD> token corresponds to index 0
    PAD_token = 0
    
    # each plot_iters display behaviour of RNN Decoder
    plot_batches = 300
    
    # gradient clipping
    clip = 10.0
    
    train_losses, train_accs = [], []
    encoder_optimizer = Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=learning_rate)
    
    n_lang_pairs = len(train_dl) * batch_size
    
    ## teacher forcing curriculum ##
    
    # decrease teacher forcing ratio per epoch or per iters respectively (start off with high ratio and move in equal steps to min_ratio)
    ratio_diff = max_ratio-min_ratio
    
    # if bs == 1, then len(train_dl) == n_iters (i.e., 100k)
    if batch_size == 1: 
        assert len(train_dl) == 100000, 'for online training, number of training examples must be set to 100k'
        assert epochs == 1, 'for online training, we run experiment for a single epoch of 100k iterations'
        decrease_every = 20000
        step_per_iters = ratio_diff / (len(train_dl) / decrease_every)
    else:
        step_per_epoch = ratio_diff / epochs
        
    teacher_forcing_ratio = max_ratio
    
    # store detailed results per epoch
    results_per_epoch = defaultdict(dict)
    
    if similarity_computation:
        command_hiddens = {}
    
    for epoch in trange(epochs,  desc="Epoch"):
        
        acc_per_epoch = 0
        losses_per_epoch = []
        
        # store detailed results for commands and actions respectively 
        results_cmds = defaultdict(dict)
        results_acts = defaultdict(dict) 
        
        for idx, (commands, input_lengths, actions, masks) in enumerate(train_dl):
                       
            commands, input_lengths, actions, masks = sort_batch(commands, input_lengths, actions, masks)
            
            losses_per_batch = []
            
            loss, n_totals = 0, 0
            
            # zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
                        
            # initialise as many hidden states as there are sequences in the mini-batch (32 in our case)
            encoder_hidden = encoder.init_hidden(batch_size)

            target_length = actions.size(1) # max_target_length
                        
            encoder_outputs, encoder_hidden = encoder(commands, input_lengths, encoder_hidden)
            
            decoder_input = actions[:, 0]
            
            # init decoder hidden with encoder's final hidden state
            if encoder.bidir:
                if hasattr(encoder, 'lstm'):
                    decoder_hidden = sum_directions(encoder_hidden, lstm=True)
                else:
                    decoder_hidden = sum_directions(encoder_hidden)
            else:
                decoder_hidden = encoder_hidden
            
            # exploit final hidden states only from last epoch (shortly before model convergence)
            if similarity_computation and epoch == epochs - 1:
                commands = commands.cpu().numpy()
                commands_str = idx_to_str_mapping(commands, i2w_source)
                command_h = decoder_hidden.squeeze(0) if decoder_hidden.size(0) == 1 else decoder_hidden[0].squeeze(0)
                for i, cmd in enumerate(commands_str):
                    # per command, store final encoder hidden states
                    command_hiddens[' '.join(cmd)] = command_h[i]

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            
            pred_sent = ""            
            preds = torch.zeros((batch_size, target_length)).to(device)
            preds[:, 0] += 1 #SOS_token
            
            if use_teacher_forcing:
                # Teacher forcing: feed target as the next input
                for i in range(1, target_length):
                    if hasattr(decoder, 'attention'):
                        decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    else:
                        decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    _, topi = decoder_out.topk(1)
                    pred = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device)
                    
                    # accumulate predictions 
                    preds[:, i] += pred

                    # calculate and accumulate loss
                    mask_loss, n_total = maskNLLLoss(decoder_out, actions[:, i], masks[:, i])
                    # mask loss is NaN towards the end, if batch only consists of sequences that are shorter than max_length
                    if not torch.isnan(mask_loss) and n_total > 0: 
                        loss += mask_loss
                        losses_per_batch.append(mask_loss.item() * n_total)
                    n_totals += n_total
                    decoder_input = actions[:, i] 
                    
                    pred_sent += i2w_target[pred[0].item()] + " "

            else:
                # Autoregressive RNN: feed previous prediction as the next input
                for i in range(1, target_length):
                    if hasattr(decoder, 'attention'):
                        decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    else:
                        decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    _, topi = decoder_out.topk(1)
                    decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device)
                    pred = decoder_input
                    
                    # accumulate predictions 
                    preds[:, i] += pred
                    mask_loss, n_total = maskNLLLoss(decoder_out, actions[:, i], masks[:, i])
                    if not torch.isnan(mask_loss) and n_total > 0:
                        loss += mask_loss
                        losses_per_batch.append(mask_loss.item() * n_total)
                    n_totals += n_total
                    
                    pred_sent += i2w_target[pred[0].item()] + " "
                    
            # skip <SOS> token and ignore <PAD> tokens
            true_sent = ' '.join([i2w_target[act.item()] for act in islice(actions[0], 1, None) if act.item() != PAD_token]).strip()
            
            # strip off any leading or trailing white spaces
            pred_sent = pred_sent.strip().split()
            pred_sent = ' '.join(pred_sent[:true_sent.split().index('<EOS>')+1])
            
            
            # update accuracy
            if detailed_results:
                results_cmds, results_acts = exact_match_accuracy_detailed(preds, actions, input_lengths, results_cmds, results_acts)
                acc_per_epoch = exact_match_accuracy(preds, actions, acc_per_epoch)
            else:
                acc_per_epoch = exact_match_accuracy(preds, actions, acc_per_epoch)
             
            loss.backward()
            
            current_loss = np.sum(losses_per_batch) / n_totals
            losses_per_epoch.append(current_loss)
            
            ### Inspect translation behaviour ###
            if detailed_analysis:
                nl_command = ' '.join([i2w_source[cmd.item()] for cmd in commands[0]]).strip()
                if idx > 0 and idx % plot_batches == 0:
                    print("Loss: {}".format(current_loss)) # current per sequence loss
                    print("Acc: {}".format(acc_per_epoch / (batch_size * (idx + 1)))) # current per iters exact-match accuracy
                    print()
                    print("Command: {}".format(nl_command))
                    print("True action: {}".format(true_sent))
                    print("Pred action: {}".format(pred_sent))
                    print()
                    print("True sent length: {}".format(len(true_sent.split())))
                    print("Pred sent length: {}".format(len(pred_sent.split())))
                    print()
                
            # clip gradients after each batch (inplace)
            _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            
            # take step
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            if batch_size == 1:
                if idx > 0 and idx % decrease_every == 0:
                    teacher_forcing_ratio -= step_per_iters
        
        if batch_size > 1:
            # compute loss and accuracy per epoch
            loss_per_epoch = np.mean(losses_per_epoch)
            acc_per_epoch /= n_lang_pairs
            
        if detailed_results:
            results_cmds = {cmd_length: (values['match'] / values['freq']) * 100 for cmd_length, values in results_cmds.items()}
            results_acts = {act_length: (values['match'] / values['freq']) * 100 for act_length, values in results_acts.items()}
            results_per_epoch['cmds'][epoch] = results_cmds
            results_per_epoch['acts'][epoch] = results_acts
            
        print("Train loss: {}".format(loss_per_epoch)) # loss
        print("Train acc: {}".format(acc_per_epoch)) # exact-match accuracy
        print("Current teacher forcing ratio {}".format(teacher_forcing_ratio))
        
        train_losses.append(loss_per_epoch)
        train_accs.append(acc_per_epoch)
        
        # decrease teacher forcing ratio per epoch
        teacher_forcing_ratio -= step_per_epoch 
    if detailed_results:
        return train_losses, train_accs, results_per_epoch, encoder, decoder
    elif similarity_computation:
        return train_losses, train_accs, command_hiddens, encoder, decoder
    else:
        return train_losses, train_accs, encoder, decoder


### Testing ###

def test(test_dl, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, batch_size:int,
         detailed_analysis:bool=True, detailed_results:bool=False, components_accuracy:bool=False):
    
    # <PAD> token corresponds to index 0
    PAD_token = 0
    
    # set models into evaluation mode
    encoder.eval()
    decoder.eval()
    
    # each n_iters plot behaviour of RNN Decoder
    plot_batches = 50
    
    # total number of language pairs
    n_lang_pairs = len(test_dl) * batch_size
    
    # NOTE: NO TEACHER FORCING DURING TESTING !!!
    
    # store detailed results for experiment 2
    if detailed_results:
        results_cmds = defaultdict(dict)
        results_acts = defaultdict(dict)
        
    if components_accuracy:
        results_per_component = defaultdict(dict)
        
    test_acc = 0
    
    # no gradient computation for evaluation mode
    with torch.no_grad():
        for idx, (commands, input_lengths, actions) in enumerate(test_dl):
            
            # if current batch_size is smaller than specified batch_size, skip batch
            if len(commands) != batch_size:
                n_lang_pairs_not_tested = len(commands)
                continue

            commands, input_lengths, actions = sort_batch(commands, input_lengths, actions, training=False)

            # initialise as many hidden states as there are sequences in the mini-batch (i.e., = batch_size)
            encoder_hidden = encoder.init_hidden(batch_size)

            target_length = actions.size(1) # max_target_length

            encoder_outputs, encoder_hidden = encoder(commands, input_lengths, encoder_hidden)

            decoder_input = actions[:, 0]
            
            # init decoder hidden with encoder's final hidden state
            if encoder.bidir:
                if hasattr(encoder, 'lstm'):
                    decoder_hidden = sum_directions(encoder_hidden, lstm=True)
                else:
                    decoder_hidden = sum_directions(encoder_hidden)
            else:
                decoder_hidden = encoder_hidden

            pred_sent = ""            
            preds = torch.zeros((batch_size, target_length)).to(device)
            preds[:, 0] += 1 #SOS_token

            # Autoregressive RNN: feed previous prediction as the next input
            for i in range(1, target_length):
                if hasattr(decoder, 'attention'):
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                _, topi = decoder_out.topk(1)
                decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device)
                pred = decoder_input
                # accumulate predictions 
                preds[:, i] += pred
                pred_sent += i2w_target[pred[0].item()] + " "

            # skip <SOS> token and ignore <PAD> tokens
            true_sent = ' '.join([i2w_target[act.item()] for act in islice(actions[0], 1, None) if act.item() != PAD_token]).strip()

            # strip off any leading or trailing white spaces
            pred_sent = pred_sent.strip().split()
            pred_sent = ' '.join(pred_sent[:true_sent.split().index('<EOS>')+1])
            
            # update accuracy
            if detailed_results:
                # accuracy as a function of command or action sequence length
                results_cmds, results_acts = exact_match_accuracy_detailed(preds, actions, input_lengths, results_cmds, results_acts)
                test_acc = exact_match_accuracy(preds, actions, test_acc)
                
            elif components_accuracy:
                # accuracy as a function of individual input components
                results_per_component = component_based_accuracy(preds, commands, i2w_source, i2w_target, results_per_component)
                test_acc = exact_match_accuracy(preds, actions, test_acc)
                
            else:
                test_acc = exact_match_accuracy(preds, actions, test_acc)

            ### Inspect translation behaviour ###
            if detailed_analysis:
                nl_command = ' '.join([i2w_source[cmd.item()] for cmd in commands[0]]).strip()
                if idx > 0 and idx % plot_batches == 0:
                    print("Current test acc: {}".format(test_acc / (batch_size * (idx + 1)))) # current per iters exact-match accuracy
                    print()
                    print("Command: {}".format(nl_command))
                    print("True action: {}".format(true_sent))
                    print("Pred action: {}".format(pred_sent))
                    print()
                    print("True sent length: {}".format(len(true_sent.split())))
                    print("Pred sent length: {}".format(len(pred_sent.split())))
                    print()
                    
    test_acc /= (n_lang_pairs - n_lang_pairs_not_tested)
    print("Test acc: {}".format(test_acc)) # exact-match test accuracy
    
    if detailed_results:
        results_cmds = {cmd_length: (vals['match'] / vals['freq']) * 100 for cmd_length, vals in results_cmds.items()}
        results_acts = {act_length: (vals['match'] / vals['freq']) * 100 for act_length, vals in results_acts.items()}
        return test_acc, results_cmds, results_acts

    elif components_accuracy:
        acc_per_component = {}
        for component, vals in results_per_component.items():
            try:
                acc_per_component[component] = (vals['match'] / vals['freq']) * 100
            # if the model is never correct for a particular phrase, there'll be no key for 'match'
            except KeyError:
                acc_per_component[component] = float(0)
        acc_per_component = dict(sorted(acc_per_component.items(), key=lambda kv:kv[1], reverse=True))
        return test_acc, acc_per_component
    
    else:
        return test_acc

### Helper functions for training and testing ###

# compute cosine similarities between hidden states of commands
def compute_similarities(command_hiddens:dict, command:str, n_neighbours:int=5):
    command_h = command_hiddens[command]
    neighbours = {}
    #del command_hiddens[command]
    for cmd, h in command_hiddens.items():
        if cmd != command:
            neighbours[cmd] = cosine_similarity(command_h, h)
    nearest_neighbours = dict(sorted(neighbours.items(), key=lambda kv:kv[1], reverse=True)[:n_neighbours])
    return nearest_neighbours

# cosine similarity
def cosine_similarity(x:torch.Tensor, y:torch.Tensor):
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    num = x @ y
    denom = np.linalg.norm(x) * np.linalg.norm(y) # default is Frobenius norm (i.e., L2 norm)
    return num / denom

# mapping of index sequences (in a batch) to corresponding string sequences
def idx_to_str_mapping(idx_seqs:list, i2w_dict:dict, special_tokens:list=[0, 1, 2]):
    return list(map(lambda idx_seq: [i2w_dict[idx] for idx in idx_seq if idx not in special_tokens], idx_seqs))

# computation of component-based (i.e., phrase-based) exact-match accuracy
def component_based_accuracy(pred_actions:torch.Tensor, commands:torch.Tensor, i2w_source:dict, i2w_target:dict,
                             results_per_component:dict, conjunctions:list=['and', 'after']):
    """phrase-based exact-match accuracy computation
    Args:
        pred_actions (torch.Tensor) - batch of predicted action sequences (i.e., yhat)
        commands (torch.Tensor) - batch of padded command sequences
        i2w_source (dict) - idx-to-word dictionary for commands (source)
        i2w_target (dict) - idx-to-word dictionary for actions (target)
        results_per_component (dict) - current exact-match performance per command component
        conjunctions (list) - list of all conjunctions in source language
    Returns:
        results_per_component (dict) - updated exact-match performance per command component
    """
    pred_actions = pred_actions.cpu().numpy().tolist()
    commands = commands.cpu().numpy().tolist()
    pred_actions_str = idx_to_str_mapping(pred_actions, i2w_target)
    commands_str = idx_to_str_mapping(commands, i2w_source)
    for cmd, pred_act in zip(commands_str, pred_actions_str):
        multiple_components = False
        for conj in conjunctions:
            if re.search(conj, ' '.join(cmd)):
                multiple_components = True
                conj_idx = cmd.index(conj)
                cmp_1 = cmd[:conj_idx]
                cmp_2 = cmd[conj_idx + 1:]
                if conj == 'and':
                    x_1 = semantic_mapping(cmp_1)
                    x_2 = semantic_mapping(cmp_2)
                    cmp_1 = ' '.join(cmp_1)
                    cmp_2 = ' '.join(cmp_2)
                    if pred_act[:len(x_1)] == x_1:
                        if 'match' in results_per_component[cmp_1]:
                            results_per_component[cmp_1]['match'] += 1
                        else:
                            results_per_component[cmp_1]['match'] = 1
                    if pred_act[len(x_1):] == x_2:
                        if 'match' in results_per_component[cmp_2]:
                            results_per_component[cmp_2]['match'] += 1
                        else:
                            results_per_component[cmp_2]['match'] = 1
                elif conj == 'after':
                    # translate in reverse command sequence order --> [x1 after x_2]
                    x_2 = semantic_mapping(cmp_1)
                    x_1 = semantic_mapping(cmp_2)
                    cmp_1 = ' '.join(cmp_1)
                    cmp_2 = ' '.join(cmp_2)
                    if pred_act[:len(x_2)] == x_2:
                        if 'match' in results_per_component[cmp_1]:
                            results_per_component[cmp_1]['match'] += 1
                        else:
                            results_per_component[cmp_1]['match'] = 1
                    if pred_act[len(x_2):] == x_1:
                        if 'match' in results_per_component[cmp_2]:
                            results_per_component[cmp_2]['match'] += 1
                        else:
                            results_per_component[cmp_2]['match'] = 1                
                if 'freq' in results_per_component[cmp_1]:
                    results_per_component[cmp_1]['freq'] += 1
                else:
                    results_per_component[cmp_1]['freq'] = 1
                if 'freq' in results_per_component[cmp_2]:
                    results_per_component[cmp_2]['freq'] += 1
                else:
                    results_per_component[cmp_2]['freq'] = 1
        if not multiple_components:
            x = semantic_mapping(cmd)
            cmd = ' '.join(cmd)
            if pred_act == x:
                if 'match' in results_per_component[cmd]:
                    results_per_component[cmd]['match'] += 1
                else:
                    results_per_component[cmd]['match'] = 1
            if 'freq' in results_per_component[cmd]: 
                results_per_component[cmd]['freq'] += 1
            else:
                results_per_component[cmd]['freq'] = 1
    return results_per_component

# this functions sums forward- and backward hidden states to leverage the full potential of bidirectional encoders
def sum_directions(encoder_hidden:torch.Tensor, lstm:bool=False):
    if lstm:
        # NOTE: this step is necessary since LSTMs contrary to RNNs and GRUs have cell states
        assert len(encoder_hidden) == 2, "LSTM's encoder hidden must consist of both hidden and cell states"
        hidden_states, cell_states = encoder_hidden
        encoder_h = torch.stack(tuple(torch.add(h, hidden_states[i-1]) 
                                      for i, h in enumerate(hidden_states) if i%2 != 0))
        encoder_c = torch.stack(tuple(torch.add(h, cell_states[i-1]) 
                                      for i, h in enumerate(cell_states) if i%2 != 0))
        return (encoder_h, encoder_c)
    else:
        return torch.stack(tuple(torch.add(h, encoder_hidden[i-1]) for i, h in enumerate(encoder_hidden) if i%2 != 0))
        
# batch sorting function (necessary for mini-batch training)
def sort_batch(commands, input_lengths, actions, masks=None, training:bool=True, pad_token:int=0):
    indices, commands = zip(*sorted(enumerate(commands.cpu().numpy()), key=lambda seq: len(seq[1][seq[1] != pad_token]), reverse=True))
    indices = np.array(list(indices))
    commands = torch.tensor(np.array(list(commands)), dtype=torch.long).to(device)
    if training:
        isinstance(masks, torch.Tensor), "tensor of token masks must be passed during training"
        return commands, input_lengths[indices], actions[indices], masks[indices]
    else:
        return commands, input_lengths[indices], actions[indices]

# exact match accuracy for mini-batch MT
def exact_match_accuracy(pred_actions:torch.Tensor, true_actions:torch.Tensor, acc:int):
    EOS_token = 2
    isinstance(acc, int)
    for pred, true in zip(pred_actions, true_actions):
        # copy tensor to CPU before converting it into a NumPy array
        pred = pred.cpu().numpy().tolist()
        true = true.cpu().numpy().tolist()
        # for each sentence, calculate exact match token accuracy (until first occurrence of an EOS token)
        try:
            pred = pred[:pred.index(EOS_token)+1]
            true = true[:true.index(EOS_token)+1]
            acc += 1 if np.array_equal(pred, true) else 0 # exact match accuracy
        except ValueError:
            continue
    return acc

# detailed exact match accuracy for command and action sequences for experiment 2
def exact_match_accuracy_detailed(pred_actions:torch.Tensor, true_actions:torch.Tensor, input_lengths:torch.Tensor,
                                  results_cmds:dict, results_acts:dict):
    EOS_token = 2
    input_lengths = list(map(lambda cmd_length: cmd_length.cpu().item(), input_lengths))
    for pred_act, true_act, cmd_length in zip(pred_actions, true_actions, input_lengths):
        
        # copy tensor to CPU before converting it into a NumPy array
        pred_act = pred_act.cpu().numpy().tolist()
        true_act = true_act.cpu().numpy().tolist()
        
        true_act = true_act[:true_act.index(EOS_token)+1]
        
        # count frequency of different command and action sequence lengths respectively
        if 'freq' in results_cmds[cmd_length]:
            results_cmds[cmd_length]['freq'] += 1
        else:
            results_cmds[cmd_length]['freq'] = 1

        if 'freq' in results_acts[len(true_act)]:
            results_acts[len(true_act)]['freq'] += 1
        else:
            results_acts[len(true_act)]['freq'] = 1
            
        # for each sentence, calculate exact match token accuracy (until first occurrence of an EOS token)
        try:
            pred_act = pred_act[:pred_act.index(EOS_token)+1]
            
            match = 1 if np.array_equal(pred_act, true_act) else 0 # exact match accuracy
            
            if 'match' in results_cmds[cmd_length]:
                results_cmds[cmd_length]['match'] += match
            else:
                results_cmds[cmd_length]['match'] = match
                
            if 'match' in results_acts[len(true_act)]:
                results_acts[len(true_act)]['match'] += match
            else:
                results_acts[len(true_act)]['match'] = match
        
        except ValueError:
            if 'match' not in results_cmds[cmd_length]:
                results_cmds[cmd_length]['match'] = 0
            else:
                pass
            if 'match' not in results_acts[len(true_act)]:
                results_acts[len(true_act)]['match'] = 0
            else:
                pass
            
    return results_cmds, results_acts

# masked negative log-likelihood loss (necessary for mini-batch training --> mask the loss for <PAD> tokens)
def maskNLLLoss(pred, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(pred, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# sampling function for experiment 1b 
def sample_distinct_pairs(cmd_act_pairs:list, ratio:float):
    # randomly shuffle the data set prior to picking distinct examples from train set
    np.random.shuffle(cmd_act_pairs)
    n_lang_pairs = len(cmd_act_pairs)
    n_distinct_samples = int(n_lang_pairs * ratio)        
    return cmd_act_pairs[:n_distinct_samples]