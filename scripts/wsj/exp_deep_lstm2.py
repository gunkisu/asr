#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import pickle
from theano import tensor as T
from collections import namedtuple

from lasagne.layers import get_all_params, count_params

from libs.deep_lstm_utils import get_arg_parser, get_save_path
from libs.utils import StopWatch, Rsync, gen_win, save_network, save_eval_history, best_fer, show_status, sync_data, \
    find_reload_model, load_or_init_model
from libs.comp_graph_utils import trainer, predictor, eval_net

from libs.lasagne_libs.utils import set_model_param_value
from libs.lasagne_libs.updates import adam
from lasagne.layers import count_params

from libs.deep_lstm_builder import build_deep_lstm
from data.fuel_utils import create_ivector_datastream

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    args.save_path = get_save_path(args)
    
    find_reload_model(args)

    print(args)
   
    if args.num_tbptt_steps:
        if args.delay >= args.num_tbptt_steps:
            print('delay >= num_tbptt_steps')
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
 
    network = build_deep_lstm(input_var=input_data,
                                mask_var=input_mask,
                                input_dim=args.input_dim,
                                num_layers=args.num_layers,
                                num_units=args.num_units,
                                num_proj_units=args.num_proj_units,
                                output_dim=args.output_dim, 
                                grad_clipping=args.grad_clipping,
                                is_bidir=not args.uni,
                                use_layer_norm=args.use_layer_norm,
                                ivector_dim=args.ivector_dim,
                                ivector_var=ivector_data)

    network_params = get_all_params(network, trainable=True)
    
    param_count = count_params(network, trainable=True)
    print('Number of parameters of the network: {:.2f}M'.format(float(param_count)/1000000))

    pretrain_update_params_val, pretrain_total_epoch_cnt = load_or_init_model(network_params, args)

    EvalRecord = namedtuple('EvalRecord', ['train_ce_frame', 'valid_ce_frame', 'valid_fer', 'test_ce_frame', 'test_fer'])
    eval_history =[EvalRecord(10.1, 10.0, 1.0, 10.0, 1.0)]

    print('Build trainer')
    sw.reset()

    training_fn, trainer_params = trainer(
              input_data=input_data,
              input_mask=input_mask,
              target_data=target_data,
              target_mask=target_mask,
              network=network,
              updater=adam,
              learning_rate=args.learn_rate,
              delay=args.delay,
              load_updater_params=pretrain_update_params_val, 
              ivector_data=ivector_data)
    
    sw.print_elapsed()

    print('Build predictor')
    sw.reset()

    predict_fn = predictor(input_data=input_data,
        input_mask=input_mask,
        target_data=target_data,
        target_mask=target_mask, 
        network=network, 
        delay=args.delay,
        ivector_data=ivector_data)

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

        for b_idx, data in enumerate(train_ds.get_epoch_iterator(), start=1):
            input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = data

            if args.use_ivector_input:
                train_output = training_fn(input_data, input_mask, ivector_data, target_data, target_mask)
            else:
                train_output = training_fn(input_data, input_mask, target_data, target_mask)

            ce_frame, network_grads_norm = train_output

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
        valid_ce_frame, valid_fer = eval_net(predict_fn, valid_ds, args.use_ivector_input, delay=args.delay)
        test_ce_frame, test_fer = eval_net(predict_fn, test_ds, args.use_ivector_input, delay=args.delay)
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

        
