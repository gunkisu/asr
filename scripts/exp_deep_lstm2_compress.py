#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import pickle
from theano import tensor as T
from collections import namedtuple

import numpy

from lasagne.layers import get_all_params, count_params

from libs.deep_lstm_utils import get_arg_parser 
from libs.utils import StopWatch, Rsync, gen_win, save_network, save_eval_history, best_fer, show_status, sync_data, \
    find_reload_model, load_or_init_model, skip_frames, compress_batch, seg_len_info
from libs.comp_graph_utils import trainer, predictor, eval_net_compress

from libs.lasagne_libs.utils import set_model_param_value
from libs.lasagne_libs.updates import adam
from lasagne.layers import count_params

from hmrnn.hmlstm_builder import build_graph_am
from hmrnn.mixer import reset_state, insert_item2dict, unzip, save_npz, save_npz2, init_tparams_with_restored_value
from libs.hmrnn_utils import add_hmrnn_graph_params

from libs.deep_lstm_builder import build_deep_lstm
from data.fuel_utils import create_ivector_datastream

if __name__ == '__main__':
    parser = get_arg_parser()
    add_hmrnn_graph_params(parser)
    parser.add_argument('model')
    args = parser.parse_args()

    args.save_path = get_save_path(args)
    
    find_reload_model(args)

    print(args)

    if args.num_tbptt_steps:
        print('--num-tbptt-steps not supported')
        sys.exit(1)

    if args.delay and not args.uni:
        print('--delay cannot be specified for bidirectional models')
        sys.exit(1)


    if args.batch_size != args.n_batch:
        print('--batch-size != --n-batch')
        sys.exit(1)
   
    sw = StopWatch()
    
    print('Loading a hmrnn model')
    f_prop, f_update, f_log_prob, f_debug, tparams, opt_tparams, \
        states, st_slope = build_graph_am(args)

    sw.print_elapsed()
    sw.reset()

    if not os.path.exists(args.model):
        print('File not found: {}'.format(args.model))
        sys.exit(1)

    tparams = init_tparams_with_restored_value(tparams, args.model)
    
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
        
        total_ce_sum = 0.0
        total_frame_count = 0
        status_sw = StopWatch()

        for b_idx, data in enumerate(train_ds.get_epoch_iterator(), start=1):
            input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = data
            n_batch, _, _ = input_data.shape
            if n_batch != args.batch_size:
                continue

            input_data_trans = numpy.transpose(input_data, (1, 0, 2))
            _, _, _, z_1_3d, _ = f_debug(input_data_trans)

            z_1_3d_trans = numpy.transpose(z_1_3d, (1,0))
            compressed_batch = [compress_batch(src, z_1_3d_trans) for src in data]
            len_info = seg_len_info(z_1_3d_trans)
            
            if args.use_ivector_input:
                train_output = training_fn(*compressed_batch)
            else:
                train_output = training_fn(compressed_batch[0], compressed_batch[1], compressed_batch[4], compressed_batch[5])

            ce_frame_sum, network_grads_norm = train_output
            total_ce_sum += ce_frame_sum
            total_frame_count += compressed_batch[-1].sum()

            if b_idx%args.log_freq == 0: 
                show_status(args.save_path, total_ce_sum / total_frame_count, network_grads_norm, b_idx, args.batch_size, e_idx)
                status_sw.print_elapsed(); status_sw.reset()
            
        print('End of Epoch {}'.format(e_idx))
        epoch_sw.print_elapsed()

        print('Saving the network')
        save_network(network_params, trainer_params, e_idx, args.save_path + '_last_model.pkl')

        print('Evaluating the network on the validation dataset')
        eval_sw = StopWatch()
        #train_ce_frame, train_fer = eval_net(predict_fn, train_ds)
        valid_ce_frame, valid_fer = eval_net_compress(predict_fn, valid_ds, f_debug, args.batch_size, args.use_ivector_input)
        test_ce_frame, test_fer = eval_net_compress(predict_fn, test_ds, f_debug, args.batch_size, args.use_ivector_input)
        eval_sw.print_elapsed()

        avg_train_ce = total_ce_sum / total_frame_count
        print('Train CE: {}'.format(avg_train_ce))
        print('Valid CE: {}, FER: {}'.format(valid_ce_frame, valid_fer))
        print('Test  CE: {}, FER: {}'.format(test_ce_frame, test_fer))
       
        if valid_fer<best_fer(eval_history):
            save_network(network_params, trainer_params, e_idx, '{}_best_model.pkl'.format(args.save_path))
        
        print('Saving the evaluation history')
        er = EvalRecord(avg_train_ce, valid_ce_frame, valid_fer, test_ce_frame, test_fer)
        eval_history.append(er)
        save_eval_history(eval_history, args.save_path + '_eval_history.pkl')

        
