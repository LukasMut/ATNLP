__all__ = ['train', 'test', 'sample_distinct_pairs']

import numpy as np
import torch.nn as nn
import random
import re
import torch

from itertools import islice
from sklearn.utils import shuffle
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import TensorDataset

from models.Encoder import *
from models.Decoder import *
from utils import pairs2idx, s2i, semantic_mapping

# set fixed random seed to reproduce results
np.random.seed(42)
random.seed(42)

device = "cpu" #("cuda" if torch.cuda.is_available() else "cpu")

### Training ###

def train(lang_pairs, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, epochs:int, batch_size:int=1,
          learning_rate:float=1e-3, detailed_analysis:bool=True, phrase_based_loss:bool=False):
    
    # number of training presentations (most training examples are shown multiple times during training, some more often than others)
    n_iters = 50000 #100000
    
    # each plot_iters display behaviour of RNN Decoder
    plot_iters = 5000 #10000
    
    # gradient clipping
    clip = 10.0
    
    train_losses, train_accs = [], []
    encoder_optimizer = Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=learning_rate)
    
    # randomly shuffle the source-target language pairs (NOTE: no need to execute the line below for experiment 1b)
    # np.random.shuffle(lang_pairs)
    
    # randomly sample 100k training pairs from the original train data set (with replacement)
    training_pairs = [pairs2idx(random.choice(lang_pairs), w2i_source, w2i_target) for _ in range(n_iters)]
    
    max_target_length = max(iter(map(lambda lang_pair: len(lang_pair[1]), training_pairs)))
    n_lang_pairs = len(training_pairs)
    
    # negative log-likelihood loss
    criterion = nn.NLLLoss()
    
    teacher_forcing_ratio = 0.5
    
    for epoch in trange(epochs,  desc="Epoch"):
                
        loss_per_epoch = 0
        acc_per_epoch = 0
        
        for idx, train_pair in enumerate(training_pairs):
            
            loss = 0
            
            command = train_pair[0].to(device)
            action = train_pair[1].to(device)
            
            # initialise as many hidden states as there are sequences in the mini-batch (1 for the beginning)
            encoder_hidden = encoder.init_hidden(batch_size)
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = command.size(0)
            target_length = action.size(0)

            encoder_outputs, encoder_hidden = encoder(command, encoder_hidden)
            
            decoder_input = action[0] # SOS token
            
            decoder_hidden = encoder_hidden # init decoder hidden with encoder hidden 

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            
            pred_sent = ""
            true_sent = ' '.join([i2w_target[act.item()] for act in islice(action, 1, None)]).strip() # skip SOS token
            
            if phrase_based_loss:
                x, multiple_components = phrase2phrase_mapping(command, i2w_source)
                length_x = len(x)
                if multiple_components:
                    length_x2 = target_length - length_x - 2 # <SOS> and <EOS> tokens must not be considered in phrase-based loss
                start_idx = 1
                decoded_phrase = []
  
            if use_teacher_forcing:
                # Teacher forcing: feed target as the next input
                for i in range(1, target_length):
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    dim = 1 if len(decoder_out.shape) > 1 else 0  # crucial to correctly compute the argmax
                    pred = torch.argmax(decoder_out, dim) # argmax computation
                    decoder_input = action[i] # convert list of int into int
                    
                    if phrase_based_loss:
                        # compute negative log-likelihood loss per individual phrase
                        decoded_phrase.append(decoder_out.squeeze(0))
                        if i == length_x and i < (target_length - 1):
                            decoded_phrase = torch.stack(tuple(out for out in decoded_phrase))
                            actual_phrase = torch.tensor(np.array([action[i].unsqueeze(0) for i in range(start_idx, length_x + 1)]))
                            loss += criterion(decoded_phrase, actual_phrase) # take the sqrt of NLLLoss per phrase (not per token)
                            decoded_phrase = []
                            if multiple_components:
                                start_idx += length_x
                                length_x += length_x2
                            loss.backward(retain_graph=True)
                                
                        elif i >= (target_length - 1):
                            loss += criterion(decoder_out, action[-1].unsqueeze(0))
                    else:
                        # compute standard negative log-likelihood loss per timestep
                        loss += criterion(decoder_out, action[i].unsqueeze(0))
                    
                    pred_sent += i2w_target[pred.item()] + " "
                    
                    if i >= (target_length - 1) and pred.squeeze().item() == w2i_target['<EOS>']:
                        break
            else:
                # Autoregressive RNN: feed previous prediction (y^t-1) as the next input
                for i in range(1, max_target_length):
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    dim = 1 if len(decoder_out.shape) > 1 else 0 # crucial to correctly compute the argmax
                    pred = torch.argmax(decoder_out, dim) # argmax computation
                                        
                    if phrase_based_loss:
                        # compute negative log-likelihood loss per individual phrase
                        decoded_phrase.append(decoder_out.squeeze(0))
                        if i == length_x and i < (target_length - 1):
                            decoded_phrase = torch.stack(tuple(out for out in decoded_phrase))
                            actual_phrase = torch.tensor(np.array([action[i].unsqueeze(0) for i in range(start_idx, length_x + 1)]))
                            loss += criterion(decoded_phrase, actual_phrase) # take the sqrt of NLLLoss per phrase (not per token)
                            decoded_phrase = []
                            if multiple_components:
                                start_idx += length_x
                                length_x += length_x2
                            loss.backward(retain_graph=True)
                                
                        elif i >= (target_length - 1):
                            loss += criterion(decoder_out, torch.tensor(w2i_target['<EOS>'],dtype=torch.long).unsqueeze(0).to(device))
                            
                    else:
                        # compute standard negative log-likelihood loss per timestep
                        if i >= (target_length - 1):
                            loss += criterion(decoder_out, torch.tensor(w2i_target['<EOS>'], dtype=torch.long).unsqueeze(0).to(device))
                        else:
                            loss += criterion(decoder_out, action[i].unsqueeze(0))
                    
                    decoder_input = pred.squeeze() # convert list of int into int
                    
                    pred_sent += i2w_target[pred.item()] + " "
                    
                    if i >= (target_length - 1) and decoder_input.item() == w2i_target['<EOS>']:
                        break
            
            pred_sent = pred_sent.strip() # strip off any leading or trailing white spaces
            acc_per_epoch += 1 if pred_sent == true_sent else 0 # exact match accuracy
            
            if phrase_based_loss:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            
            ### Inspect translation behaviour ###
            if detailed_analysis:
                nl_command = ' '.join([i2w_source[cmd.item()] for cmd in command]).strip()
                if idx > 0 and idx % plot_iters == 0:
                    print("Loss: {}".format(loss.item() / target_length)) # current per sequence loss
                    print("Acc: {}".format(acc_per_epoch / (idx + 1))) # current per iters exact-match accuracy
                    print()
                    print("Command: {}".format(nl_command))
                    print("True action: {}".format(true_sent))
                    print("Pred action: {}".format(pred_sent))
                    print()
                    print("True sent length: {}".format(len(true_sent.split())))
                    print("Pred sent length: {}".format(len(pred_sent.split())))
                    print()
            
            # clip gradients after each iteration / presentation (inplace)
            _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            
            encoder_optimizer.step()
            decoder_optimizer.step()

            loss_per_epoch += loss.item() / target_length
        
        loss_per_epoch /= n_lang_pairs
        acc_per_epoch /= n_lang_pairs
        
        print("Train loss: {}".format(loss_per_epoch)) # loss
        print("Train acc: {}".format(acc_per_epoch)) # exact-match accuracy
        print("Current teacher forcing ratio {}".format(teacher_forcing_ratio))
        
        train_losses.append(loss_per_epoch)
        train_accs.append(acc_per_epoch)
                
    return train_losses, train_accs, encoder, decoder


### Testing ###

def test(lang_pairs, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, batch_size:int=1, detailed_analysis:bool=True):
    
    # each n_iters plot behaviour of RNN Decoder
    plot_iters = 1000
    
    # randomly shuffle our source-target language pairs
    np.random.shuffle(lang_pairs)
    test_pairs = [pairs2idx(lang_pair, w2i_source, w2i_target) for lang_pair in lang_pairs]
    
    max_target_length = max(iter(map(lambda lang_pair: len(lang_pair[1]), test_pairs)))
    n_lang_pairs = len(test_pairs)
    
    # NOTE: NO TEACHER FORCING DURING TESTING !!!
                    
    test_acc = 0

    for idx, test_pair in enumerate(test_pairs):

        command = test_pair[0].to(device)
        action = test_pair[1].to(device)

        # initialise as many hidden states as there are sequences in the mini-batch (1 for the beginning)
        encoder_hidden = encoder.init_hidden(batch_size)

        input_length = command.size(0)
        target_length = action.size(0)

        encoder_outputs, encoder_hidden = encoder(command, encoder_hidden)

        decoder_input = action[0] # SOS token

        decoder_hidden = encoder_hidden # init decoder hidden with encoder hidden 

        pred_sent = ""
        true_sent = ' '.join([i2w_target[act.item()] for act in islice(action, 1, None)]).strip() # skip SOS token

        # Autoregressive RNN: feed previous prediction as the next input
        for i in range(1, max_target_length):
            decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
            dim = 1 if len(decoder_out.shape) > 1 else 0 # crucial to correctly compute the argmax
            pred = torch.argmax(decoder_out, dim) # argmax computation

            decoder_input = pred.squeeze() # convert list of int into int

            pred_sent += i2w_target[pred.item()] + " "

            if decoder_input.item() == w2i_target['<EOS>']:
                break

        # strip off any leading or trailing white spaces
        pred_sent = pred_sent.strip()
        test_acc += 1 if pred_sent == true_sent else 0 # exact match accuracy

        ### Inspect translation behaviour ###
        if detailed_analysis:
            nl_command = ' '.join([i2w_source[cmd.item()] for cmd in command]).strip()
            if idx > 0 and idx % plot_iters == 0:
                print("Test acc: {}".format(test_acc / (idx + 1))) # current per iters exact-match accuracy
                print()
                print("Command: {}".format(nl_command))
                print("True action: {}".format(true_sent))
                print("Pred action: {}".format(pred_sent))
                print()
                print("True sent length: {}".format(len(true_sent.split())))
                print("Pred sent length: {}".format(len(pred_sent.split())))
                print()
                
    test_acc /= n_lang_pairs
    print("Test acc: {}".format(test_acc)) # exact-match test accuracy
    return test_acc


### helper functions (e.g., to compute phrase-based loss) ###

# command-component to corresponding action-component semantic mapping function
def phrase2phrase_mapping(command:torch.Tensor, i2w_source:dict):
    conjunctions = ['and', 'after']
    command = command.cpu().numpy().tolist()
    command_str = idx_to_str_mapping(command, i2w_source)
    multiple_components = False
    for conj in conjunctions:
        if re.search(conj, ' '.join(command_str)):
            multiple_components = True
            conj_idx = command_str.index(conj)
            if conj == 'after':
                x = semantic_mapping(command_str[conj_idx + 1:])
            else:
                x = semantic_mapping(command_str[:conj_idx])
    if not multiple_components:
        x = semantic_mapping(command_str)   
    return x, multiple_components

# mapping of index sequences (in a batch) to corresponding string sequences
def idx_to_str_mapping(idx_seq:list, i2w_dict:dict, special_tokens:list=[0, 1, 2]):
    return [i2w_dict[idx] for idx in idx_seq if idx not in special_tokens]

# sampling function for experiment 1b
def sample_distinct_pairs(cmd_act_pairs:list, ratio:float):
    # randomly shuffle the data set prior to picking distinct examples
    np.random.shuffle(cmd_act_pairs)
    n_lang_pairs = len(cmd_act_pairs)
    n_distinct_samples = int(n_lang_pairs * ratio)        
    return cmd_act_pairs[:n_distinct_samples]