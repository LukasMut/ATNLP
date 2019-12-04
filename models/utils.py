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

def sort_batch(commands, input_lengths, actions, masks):
    indices, commands = zip(*sorted(enumerate(commands.cpu().numpy()), key=lambda seq: len(seq[1][seq[1] != 0]), reverse=True))
    indices = np.array(list(indices))
    commands = torch.tensor(np.array(list(commands)), dtype=torch.long).to(device)
    input_lengths = input_lengths[indices]
    actions = actions[indices]
    masks = masks[indices]
    return commands, input_lengths, actions, masks

### Training ###

def train(train_dl, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, epochs:int, batch_size:int,
          learning_rate:float=1e-3, max_ratio:float=0.95, min_ratio:float=0.15, detailed_analysis:bool=True):
        
    # each plot_iters display behaviour of RNN Decoder
    plot_batches = 300
    
    # gradient clipping
    clip = 10.0
    
    train_losses, train_accs = [], []
    encoder_optimizer = Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=learning_rate)
    
    n_lang_pairs = len(train_dl) * batch_size
    
    # teacher forcing curriculum
    # decrease teacher forcing ratio per epoch (start off with high ratio and move in equal steps to min_ratio)
    ratio_diff = max_ratio-min_ratio
    step_per_epoch = ratio_diff / epochs
    teacher_forcing_ratio = max_ratio
    
    for epoch in trange(epochs,  desc="Epoch"):
        
        train_batch_losses = []
        acc_per_epoch = 0
        
        for idx, (commands, input_lengths, actions, masks) in enumerate(train_dl):
           
            commands, input_lengths, actions, masks = sort_batch(commands, input_lengths, actions, masks)
            
            # zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
           
            loss, n_totals = 0, 0
                        
            # initialise as many hidden states as there are sequences in the mini-batch (1 for the beginning)
            encoder_hidden = encoder.init_hidden(batch_size)

            target_length = actions.size(1) # max_target_length
                        
            encoder_outputs, encoder_hidden = encoder(commands, input_lengths, encoder_hidden)
            
            decoder_input = actions[:, 0]
            
            decoder_hidden = encoder_hidden[:decoder.n_layers] # init decoder hidden with encoder hidden 

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            pred_sent = ""
            true_sent = ' '.join([i2w_target[act.item()] for act in islice(actions[0], 1, None)]).strip() # skip SOS token
            
            preds = torch.zeros((batch_size, target_length)).to(device)
            preds[:, 0] += 1 #SOS_token
            
            if use_teacher_forcing:
                # Teacher forcing: feed target as the next input
                for i in range(1, target_length):
                    
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)

                    ## NOTE: two lines below only work for mini batch size = 1 ##
                   
                    #dim = 1 if len(decoder_out.shape) > 1 else 0  # crucial to correctly compute the argmax
                    #pred = torch.argmax(decoder_out, dim).to(device) # argmax computation 
                    
                    _, topi = decoder_out.topk(1)
                    
                    pred = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device)
                    
                    # accumulate predictions 
                    preds[:, i] += pred

                    # calculate and accumulate loss
                    mask_loss, n_total = maskNLLLoss(decoder_out, actions[:, i], masks[:, i])
                    loss += mask_loss
                    train_batch_losses.append(mask_loss.item() * n_total)
                    n_totals += n_total
                    
                    decoder_input = actions[:, i]
                    
                    
                    pred_sent += i2w_target[pred[0].item()] + " "

            else:
                # Autoregressive RNN: feed previous prediction as the next input
                for i in range(1, target_length):
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    
                    ## NOTE: two lines below only work for mini batch size = 1 ##
                        
                    #dim = 1 if len(decoder_out.shape) > 1 else 0 # crucial to correctly compute the argmax
                    #pred = torch.argmax(decoder_out, dim).to(device) # argmax computation
                    
                    _, topi = decoder_out.topk(1)
                    
                    decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device)
                    pred = decoder_input
                    
                    # accumulate predictions 
                    preds[:, i] += pred
                    
                    mask_loss, n_total = maskNLLLoss(decoder_out, actions[:, i], masks[:, i])
                    loss += mask_loss
                    train_batch_losses.append(mask_loss.item() * n_total)
                    n_totals += n_total
                    
                    pred_sent += i2w_target[pred[0].item()] + " "
            
            # strip off any leading or trailing white spaces
            pred_sent = pred_sent.strip()
            
            for pred, true in zip(preds, actions):
                # copy tensor to CPU before converting it into a NumPy array
                acc_per_epoch += 1 if np.array_equal(pred.cpu().numpy(), true.cpu().numpy()) else 0 # exact match accuracy
        
            loss.backward()
            
            ### Inspect translation behaviour ###
            if detailed_analysis:
                nl_command = ' '.join([i2w_source[cmd.item()] for cmd in commands[0]]).strip()
                if idx > 0 and idx % plot_batches == 0:
                    print("Loss: {}".format(np.sum(train_batch_losses) / n_totals)) # current per sequence loss
                    print("Acc: {}".format(acc_per_epoch / (idx + 1) * batch_size)) # current per iters exact-match accuracy
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
        
        loss_per_epoch = np.sum(train_batch_losses) / n_totals
        acc_per_epoch /= n_lang_pairs
        
        print("Train loss: {}".format(loss_per_epoch)) # loss
        print("Train acc: {}".format(acc_per_epoch)) # exact-match accuracy
        print("Current teacher forcing ratio {}".format(teacher_forcing_ratio))
        
        train_losses.append(loss_per_epoch)
        train_accs.append(acc_per_epoch)
        
        teacher_forcing_ratio -= step_per_epoch # decrease teacher forcing ratio
        
    return train_losses, train_accs, encoder, decoder


### Testing ###

def test(test_dl, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, batch_size:int=1, detailed_analysis:bool=True):
    
    # each n_iters plot behaviour of RNN Decoder
    plot_iters = 1000
        
    n_lang_pairs = len(test_dl) * batch_size
    
    # NOTE: NO TEACHER FORCING DURING TESTING !!!
                    
    test_acc = 0

    for idx, (command, action) in enumerate(test_dl):

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
        for i in range(1, target_length):
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