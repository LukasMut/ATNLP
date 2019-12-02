__all__ = ['train', 'test', 'sample_distinct_pairs']

import numpy as np
import torch.nn as nn
import random
import torch

from itertools import islice
from sklearn.utils import shuffle
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import TensorDataset

from models.Encoder import *
from models.Decoder import *
from utils import pairs2idx, s2i

# set fixed random seed to reproduce results
np.random.seed(42)
random.seed(42)

device = ("cuda" if torch.cuda.is_available() else "cpu")

### Training ###

def train(lang_pairs, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, epochs:int, batch_size:int=1,
          learning_rate:float=1e-3, max_ratio:float=0.95, min_ratio:float=0.15, detailed_analysis:bool=True):
    
    # number of training presentations (most training examples are shown multiple times during training, some more often than others)
    n_iters = 100000
    
    # each plot_iters display behaviour of RNN Decoder
    plot_iters = 10000
    
    train_losses, train_accs = [], []
    encoder_optimizer = Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=learning_rate)
    
    # randomly shuffle the source-target language pairs (NOTE: no need to execute the line below for experiment 1b)
    # np.random.shuffle(lang_pairs)
    
    # randomly sample 100k training pairs from the original train data set (with replacement)
    #training_pairs = [pairs2idx(random.choice(lang_pairs), w2i_source, w2i_target) for _ in range(n_iters)]
    
    max_target_length = max(iter(map(lambda lang_pair: len(lang_pair[1]), training_pairs)))
    n_lang_pairs = len(training_pairs)
    
    # negative log-likelihood loss
    criterion = nn.NLLLoss()
    
    # teacher forcing curriculum
    # decrease teacher forcing ratio per epoch (start off with high ratio and move in equal steps to min_ratio)
    ratio_diff = max_ratio-min_ratio
    step_per_epoch = ratio_diff / epochs
    teacher_forcing_ratio = max_ratio
    
    for epoch in trange(epochs,  desc="Epoch"):
                
        loss_per_epoch = 0
        acc_per_epoch = 0
        
        for idx, source, target in enumerate(train_dl):
            
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

            if use_teacher_forcing:
                # Teacher forcing: feed target as the next input
                for i in range(1, target_length):
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    dim = 1 if len(decoder_out.shape) > 1 else 0  # crucial to correctly compute the argmax
                    pred = torch.argmax(decoder_out, dim) # argmax computation
                    
                    loss += criterion(decoder_out, action[i].unsqueeze(0))
                    decoder_input = action[i] # convert list of int into int
                    
                    pred_sent += i2w_target[pred.item()] + " "
                    
                    if pred.squeeze().item() == w2i_target['<EOS>']:
                        break
            else:
                # Autoregressive RNN: feed previous prediction as the next input
                for i in range(1, max_target_length):
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    dim = 1 if len(decoder_out.shape) > 1 else 0 # crucial to correctly compute the argmax
                    pred = torch.argmax(decoder_out, dim) # argmax computation
                    
                    if i >= target_length:
                        loss += criterion(decoder_out, torch.tensor(w2i_target['<EOS>'], dtype=torch.long).unsqueeze(0).to(device))
                    else:
                        loss += criterion(decoder_out, action[i].unsqueeze(0))
                    
                    decoder_input = pred.squeeze() # convert list of int into int
                    
                    pred_sent += i2w_target[pred.item()] + " "
                    
                    if decoder_input.item() == w2i_target['<EOS>']:
                        break
            
            # strip off any leading or trailing white spaces
            pred_sent = pred_sent.strip()
            acc_per_epoch += 1 if pred_sent == true_sent else 0 # exact match accuracy
        
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
        
        teacher_forcing_ratio -= step_per_epoch # decrease teacher forcing ratio
        
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

# masked negative log-likelihood loss (necessary for mini-batch training)

def maskNLLLoss(pred, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(pred, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

## Sampling function for experiment 1b ##

def sample_distinct_pairs(cmd_act_pairs:list, ratio:float):
    # randomly shuffle the data set prior to picking distinct examples
    np.random.shuffle(cmd_act_pairs)
    n_lang_pairs = len(cmd_act_pairs)
    n_distinct_samples = int(n_lang_pairs * ratio)        
    return cmd_act_pairs[:n_distinct_samples]