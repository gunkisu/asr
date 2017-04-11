#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import pickle
from theano import tensor as T
from collections import namedtuple
import math

from lasagne.layers import get_all_params, count_params

from libs.deep_lstm_utils import get_arg_parser, get_save_path
from libs.utils import StopWatch, Rsync, gen_win, save_network, save_eval_history, best_fer, show_status, sync_data, \
    find_reload_model, load_or_init_model, print_param_count
from libs.comp_graph_utils import trainer_tbptt, predictor_tbptt, eval_net_tbptt

from libs.lasagne_libs.updates import adam

from libs.deep_lstm_builder import build_deep_lstm_tbptt
from data.fuel_utils import create_ivector_datastream

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    args.save_path = get_save_path(args)

    find_reload_model(args)

    print(args)
   
    if not args.num_tbptt_steps:
        print('You must specify --num-tbptt-steps')
        sys.exit(1)

    sw = StopWatch()

    print('Build and compile network')
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    ivector_data = None
    if args.use_ivector_input:
        ivector_data = T.ftensor3('ivector_data')
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')
    is_first_win = T.iscalar('is_first_win')

    network, tbptt_layers = build_deep_lstm_tbptt(
                                input_var=input_data,
                                mask_var=input_mask,
                                input_dim=args.input_dim,
                                num_layers=args.num_layers,
                                num_units=args.num_units,
                                num_proj_units=args.num_proj_units,
                                output_dim=args.output_dim, 
                                batch_size=args.batch_size,
                                grad_clipping=args.grad_clipping,
                                is_bidir=not args.uni,
                                use_layer_norm=args.use_layer_norm,
                                ivector_dim=args.ivector_dim,
                                ivector_var=ivector_data)

    network_params = get_all_params(network, trainable=True)

    print_param_count(network)
    
    pretrain_update_params_val, pretrain_total_epoch_cnt = load_or_init_model(network_params, args)

    EvalRecord = namedtuple('EvalRecord', ['train_ce_frame', 'valid_ce_frame', 'valid_fer', 'test_ce_frame', 'test_fer'])
    eval_history =[EvalRecord(10.1, 10.0, 1.0, 10.0, 1.0)]

    sw.reset()
    print('Building trainer')
    training_fn, trainer_params = trainer_tbptt(
              input_data=input_data,
              input_mask=input_mask,
              target_data=target_data,
              target_mask=target_mask,
              network=network,
              updater=adam,
              learning_rate=args.learn_rate,
              tbptt_layers=tbptt_layers, 
              is_first_win=is_first_win,
              load_updater_params=pretrain_update_params_val, 
              ivector_data=ivector_data, delay=args.delay)
    sw.print_elapsed()

    sw.reset()
    print('Building predictor')
    predict_fn = predictor_tbptt(
        input_data=input_data,
        input_mask=input_mask,
        target_data=target_data,
        target_mask=target_mask, 
        network=network, 
        tbptt_layers=tbptt_layers, 
        is_first_win=is_first_win,
        ivector_data=ivector_data, delay=args.delay)
    sw.print_elapsed()

    print('Load data streams from {}'.format(args.train_dataset))
    sync_data(args)
    datasets = [args.train_dataset, args.valid_dataset, args.test_dataset]
    train_ds, valid_ds, test_ds = [create_ivector_datastream(path=args.data_path, which_set=dataset, 
        batch_size=args.batch_size, delay=args.delay) for dataset in datasets]

    print('Start training')

    # e_idx starts from 1
    for e_idx in range(1, args.num_epochs+1):
        if e_idx <= pretrain_total_epoch_cnt:
            print('Skip Epoch {}'.format(e_idx))
            continue

        epoch_sw = StopWatch()
        print('--')
        print('Epoch {} starts'.format(e_idx))
        print('--')
        
        train_ce_frame_sum = 0.0
        status_sw = StopWatch()

        for b_idx, batch in enumerate(train_ds.get_epoch_iterator(), start=1):
            for l in tbptt_layers:
                l.reset()

            i_data, _, _, _, _, t_mask = batch
            n_batch, _, _ = i_data.shape
            if n_batch < args.batch_size: continue

            ce_frame = 0.0
            network_grads_norm = 0.0
            for win_idx, win in enumerate(gen_win(batch, args.num_tbptt_steps), start=1):
                input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = win
                is_first_win = 1 if win_idx == 1 else 0

                if args.use_ivector_input:
                    train_output = training_fn(input_data, input_mask, ivector_data, target_data, target_mask, is_first_win)
                else:
                    train_output = training_fn(input_data, input_mask, target_data, target_mask, is_first_win)

                ce_frame_sum, network_grads_norm_sum = train_output

                ce_frame += ce_frame_sum
                network_grads_norm_sum += network_grads_norm_sum

            ce_frame = ce_frame / t_mask[:,args.delay:].sum()
            network_grads_norm = math.sqrt(network_grads_norm_sum)

            if b_idx%args.log_freq == 0: 
                show_status(args.save_path, ce_frame, network_grads_norm, b_idx, args.batch_size, e_idx)
                status_sw.print_elapsed(); status_sw.reset()
            train_ce_frame_sum += ce_frame

        print('End of Epoch {}'.format(e_idx))
        epoch_sw.print_elapsed()

        print('Saving the network')
        save_network(network_params, trainer_params, e_idx, args.save_path + '_last_model.pkl')

        print('Evaluating the network on the validation dataset')
        eval_sw = StopWatch()
        #train_ce_frame, train_fer = eval_net(predict_fn, train_ds)
        valid_ce_frame, valid_fer = eval_net_tbptt(predict_fn, valid_ds, tbptt_layers, args.num_tbptt_steps, 
            args.batch_size, args.use_ivector_input, delay=args.delay)
        test_ce_frame, test_fer = eval_net_tbptt(predict_fn, test_ds, tbptt_layers, args.num_tbptt_steps, 
            args.batch_size, args.use_ivector_input, delay=args.delay)
        eval_sw.print_elapsed()

        print('Train CE: {}'.format(train_ce_frame_sum / b_idx))
        print('Valid CE: {}, FER: {}'.format(valid_ce_frame, valid_fer))
        print('Test  CE: {}, FER: {}'.format(test_ce_frame, test_fer))
       
        if valid_fer<best_fer(eval_history):
            save_network(network_params, trainer_params, e_idx, '{}_best_model.pkl'.format(args.save_path))
        
        print('Saving the evaluation history')
        er = EvalRecord(train_ce_frame_sum / b_idx, valid_ce_frame, valid_fer, test_ce_frame, test_fer)
        eval_history.append(er)
        save_eval_history(eval_history, args.save_path + '_eval_history.pkl')

        
