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

# set fixed random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = ("cuda" if torch.cuda.is_available() else "cpu")
    
### Training ###

def train(train_dl, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, epochs:int, batch_size:int,
          learning_rate:float=1e-3, max_ratio:float=0.95, min_ratio:float=0.15, detailed_analysis:bool=True):
    
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
    
    # decrease teacher forcing ratio per epoch (start off with high ratio and move in equal steps to min_ratio)
    ratio_diff = max_ratio-min_ratio
    step_per_epoch = ratio_diff / epochs
    teacher_forcing_ratio = max_ratio
    
    for epoch in trange(epochs,  desc="Epoch"):
        
        acc_per_epoch = 0
        losses_per_epoch = []
        
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
            
            # init decoder hidden with encoder's final hidden state (only necessary for bidirectional encoders)
            if hasattr(encoder, 'lstm'):
                # NOTE: this step is necessary since LSTMs contrary to RNNs and GRUs have cell states
                decoder_hidden = tuple(hidden[:decoder.n_layers] for hidden in encoder_hidden)
            else:
                decoder_hidden = encoder_hidden[:decoder.n_layers]

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            
            pred_sent = ""            
            preds = torch.zeros((batch_size, target_length)).to(device)
            preds[:, 0] += 1 #SOS_token
            
            if use_teacher_forcing:
                # Teacher forcing: feed target as the next input
                for i in range(1, target_length):
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
        
        # compute loss and accuracy per epoch
        loss_per_epoch = np.mean(losses_per_epoch)
        acc_per_epoch /= n_lang_pairs
        
        print("Train loss: {}".format(loss_per_epoch)) # loss
        print("Train acc: {}".format(acc_per_epoch)) # exact-match accuracy
        print("Current teacher forcing ratio {}".format(teacher_forcing_ratio))
        
        train_losses.append(loss_per_epoch)
        train_accs.append(acc_per_epoch)
        
        # decrease teacher forcing ratio per epoch
        teacher_forcing_ratio -= step_per_epoch 
        
    return train_losses, train_accs, encoder, decoder


### Testing ###

def test(test_dl, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, batch_size:int, detailed_analysis:bool=True):
    
    # set models into evaluation mode
    encoder.eval()
    decoder.eval()
    
    # each n_iters plot behaviour of RNN Decoder
    plot_batches = 50
    # total number of language pairs
    n_lang_pairs = len(test_dl) * batch_size
    
    # NOTE: NO TEACHER FORCING DURING TESTING !!!
                    
    test_acc = 0
    
    # no gradient computation for evaluation mode
    with torch.no_grad():
        for idx, (commands, input_lengths, actions) in enumerate(test_dl):
            # if current batch_size is smaller than batch_size, skip batch
            if len(commands) != batch_size:
                n_lang_pairs_not_tested = len(commands)
                continue

            commands, input_lengths, actions = sort_batch(commands, input_lengths, actions, training=False)

            # initialise as many hidden states as there are sequences in the mini-batch (i.e., = batch_size)
            encoder_hidden = encoder.init_hidden(batch_size)

            target_length = actions.size(1) # max_target_length

            encoder_outputs, encoder_hidden = encoder(commands, input_lengths, encoder_hidden)

            decoder_input = actions[:, 0]

            # init decoder hidden with encoder's final hidden state (necessary for bidirectional encoders)
            if hasattr(encoder, 'lstm'):
                # NOTE: this step is necessary since LSTMs contrary to RNNs and GRUs have cell states
                decoder_hidden = tuple(hidden[:decoder.n_layers] for hidden in encoder_hidden)
            else:
                decoder_hidden = encoder_hidden[:decoder.n_layers]

            pred_sent = ""            
            preds = torch.zeros((batch_size, target_length)).to(device)
            preds[:, 0] += 1 #SOS_token

            # Autoregressive RNN: feed previous prediction as the next input
            for i in range(1, target_length):
                decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                _, topi = decoder_out.topk(1)
                decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device)
                pred = decoder_input
                # accumulate predictions 
                preds[:, i] += pred
                pred_sent += i2w_target[pred[0].item()] + " "

            # skip <SOS> token and ignore <PAD> tokens
            true_sent = ' '.join([i2w_target[act.item()] for act in islice(actions[0], 1, None) if act.item() != 0]).strip()

            # strip off any leading or trailing white spaces
            pred_sent = pred_sent.strip().split()
            pred_sent = ' '.join(pred_sent[:true_sent.split().index('<EOS>')+1])
            
            # update accuracy
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
    return test_acc

### Helper functions for training and testing ###

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

# masked negative log-likelihood loss (necessary for mini-batch training)

def maskNLLLoss(pred, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(pred, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# sampling function for experiment 1b #

def sample_distinct_pairs(cmd_act_pairs:list, ratio:float):
    # randomly shuffle the data set prior to picking distinct examples from train set
    np.random.shuffle(cmd_act_pairs)
    n_lang_pairs = len(cmd_act_pairs)
    n_distinct_samples = int(n_lang_pairs * ratio)        
    return cmd_act_pairs[:n_distinct_samples]