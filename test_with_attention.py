def test(test_dl, w2i_source, w2i_target, i2w_source, i2w_target, encoder, decoder, batch_size:int,
         detailed_analysis:bool=True, detailed_results:bool=False, plot_attn:bool=False):
    
    # <PAD> token corresponds to index 0
    PAD_token = 0
    
    # set models into evaluation mode
    encoder.eval()
    decoder.eval()
    
    # each n_iters plot behaviour of RNN Decoder
    plot_batches = 23
    # total number of language pairs
    n_lang_pairs = len(test_dl) * batch_size
    
    # NOTE: NO TEACHER FORCING DURING TESTING !!!
    
    # store detailed results for experiment 2
    results_cmds = defaultdict(dict)
    results_acts = defaultdict(dict) 
    
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
            attn_wgt_mat = []
            # Autoregressive RNN: feed previous prediction as the next input
            for i in range(1, target_length):
                if hasattr(decoder, 'attention'):
                    decoder_out, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                _, topi = decoder_out.topk(1)
                decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device)
                pred = decoder_input
                # accumulate predictions 
                preds[:, i] += pred
                pred_sent += i2w_target[pred[0].item()] + " "
                attn_wgt_mat.append(attn_weights[0].squeeze().cpu().numpy())

            # skip <SOS> token and ignore <PAD> tokens
            true_sent = ' '.join([i2w_target[act.item()] for act in islice(actions[0], 1, None) if act.item() != PAD_token]).strip()

            # strip off any leading or trailing white spaces
            pred_sent = pred_sent.strip().split()
            pred_sent = ' '.join(pred_sent[:true_sent.split().index('<EOS>')+1])
            
            # update accuracy
            if detailed_results:
                results_cmds, results_acts = exact_match_accuracy_detailed(preds, actions, input_lengths, results_cmds, results_acts)
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
                    if plot_attn:
                        t = np.array(attn_wgt_mat)
                        t = t[:len(pred_sent.split()),:]
                        ax = sns.heatmap(t)
                        ax.set_xticklabels(nl_command.split(), rotation=90)
                        ax.set_yticklabels(pred_sent.split(), rotation=0)
                        bottom, top = ax.get_ylim()
                        left, right = ax.get_xlim()
                        ax.set_xlim(0,len(nl_command.split()))
                        ax.set_ylim(len(pred_sent.split()),0)
                        print(right)
                        plt.show(ax)
                    
    test_acc /= (n_lang_pairs - n_lang_pairs_not_tested)
    print("Test acc: {}".format(test_acc)) # exact-match test accuracy
    
    if detailed_results:
        results_cmds = {cmd_length: (value['match'] / value['frequency']) * 100 for cmd_length, value in results_cmds.items()}
        results_acts = {act_length: (value['match'] / value['frequency']) * 100 for act_length, value in results_acts.items()}
        return test_acc, results_cmds, results_acts
    else:
        return test_acc

