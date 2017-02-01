#!/usr/bin/env python
from __future__ import print_function

import os
import pickle
from theano import tensor as T
from collections import namedtuple

from libs.hyperlstm_utils import add_params, get_arg_parser, get_save_path
from libs.lasagne_libs.utils import set_model_param_value

from lasagne.layers import get_all_params, count_params
from libs.lasagne_libs.updates import momentum

from libs.comp_graph_utils import trainer, predictor, eval_net
from libs.utils import save_network, save_eval_history, best_fer, show_status
from libs.utils import StopWatch, Rsync
from models.deep_hyperlstm import build_deep_hyperlstm

from data.wsj.fuel_utils import get_datastream

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    args.save_path = get_save_path(args)
    
    if not args.reload_model:
        reload_path = args.save_path + '_last_model.pkl'

        if os.path.exists(reload_path):
            print('Previously trained model detected: {}'.format(reload_path))
            print('Training continues')
            args.reload_model = reload_path

    print(args)

    sw = StopWatch()
       
    print('Loading data streams from {}'.format(args.data_path))
    if not args.no_copy:
        print('Copying data to local machine...')
        rsync = Rsync(args.tmpdir)
        rsync.sync(args.data_path)
        args.data_path = os.path.join(args.tmpdir, os.path.basename(args.data_path))
        sw.print_elapsed()

    train_ds = get_datastream(path=args.data_path,
                                  which_set=args.train_dataset,
                                  batch_size=args.batch_size, 
                                  use_ivectors=args.use_ivectors, 
                                  truncate_ivectors=args.truncate_ivectors, 
                                  ivector_dim=args.ivector_dim)
    valid_ds = get_datastream(path=args.data_path,
                                  which_set=args.valid_dataset,
                                  batch_size=args.batch_size, 
                                  use_ivectors=args.use_ivectors,
                                  truncate_ivectors=args.truncate_ivectors,
                                  ivector_dim=args.ivector_dim)
    test_ds = get_datastream(path=args.data_path,
                                  which_set=args.test_dataset,
                                  batch_size=args.batch_size, 
                                  use_ivectors=args.use_ivectors,
                                  truncate_ivectors=args.truncate_ivectors,
                                  ivector_dim=args.ivector_dim)
    

    print('Building and compiling network')
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')
    
    network = build_deep_hyperlstm(input_var=input_data,
                             mask_var=input_mask,
                             input_dim=args.input_dim,
                             num_layers=args.num_layers,
                             num_units=args.num_nodes,
                             num_hyper_units=args.num_hyper_nodes,
                             num_proj_units=args.num_proj_nodes,
                             output_dim=args.output_dim,
                             grad_clipping=args.grad_clipping,
                             bidir=not args.unidirectional)

    network_params = get_all_params(network, trainable=True)
    param_count = count_params(network, trainable=True)
    print('Number of parameters of the network: {:.2f}M'.format(float(param_count)/1000000))

    EvalRecord = namedtuple('EvalRecord', ['train_ce_frame', 'valid_ce_frame', 'valid_fer', 'test_ce_frame', 'test_fer'])

    if args.reload_model:
        print('Loading model: {}'.format(args.reload_model))
        with open(args.reload_model, 'rb') as f:
            pretrain_network_params_val,  pretrain_update_params_val, \
                    pretrain_total_epoch_cnt = pickle.load(f)

        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        pretrain_update_params_val = None
        pretrain_total_epoch_cnt = 0
    
    eval_history =[EvalRecord(10.1, 10.0, 1.0, 10.0, 1.0)]

    print('Building trainer')
    sw.reset()
    training_fn, trainer_params = trainer(
              input_data=input_data,
              input_mask=input_mask,
              target_data=target_data,
              target_mask=target_mask,
              num_outputs=args.output_dim,
              network=network,
              updater=eval(args.updater),
              learning_rate=args.learn_rate,
              load_updater_params=pretrain_update_params_val)
    sw.print_elapsed()

    print('Building predictor')
    sw.reset()
    predict_fn = predictor(input_data=input_data,
                                       input_mask=input_mask,
                                       target_data=target_data,
                                       target_mask=target_mask,
                                       num_outputs=args.output_dim,
                                       network=network)

    sw.print_elapsed()
    print('Starting')

    # e_idx starts from 1
    for e_idx in range(1, args.num_epochs+1):
        if e_idx <= pretrain_total_epoch_cnt:
            print('Skipping Epoch {}'.format(e_idx))
            continue

        epoch_sw = StopWatch()
        print('--')
        print('Epoch {} starts'.format(e_idx))
        print('--')
        
        train_ce_frame_sum = 0.0
        status_sw = StopWatch()
        for b_idx, data in enumerate(train_ds.get_epoch_iterator(), start=1):
            input_data, input_mask, target_data, target_mask = data
            train_output = training_fn(input_data,
                                       input_mask,
                                       target_data,
                                       target_mask)
            ce_frame = train_output[0]
            network_grads_norm = train_output[1]

            if b_idx%args.train_disp_freq == 0: 
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
        valid_ce_frame, valid_fer = eval_net(predict_fn, valid_ds)
        test_ce_frame, test_fer = eval_net(predict_fn, test_ds)
        eval_sw.print_elapsed()

        print('Train CE: {}'.format(train_ce_frame_sum / b_idx))
        print('Valid CE: {}, FER: {}'.format(valid_ce_frame, valid_fer))
        print('Test  CE: {}, FER: {}'.format(test_ce_frame, test_fer))
       
        if valid_fer<best_fer(eval_history):
            symlink_force('{}_last_model.pkl'.format(args.save_path), '{}_best_model.pkl'.format(args.save_path)) 
        
        print('Saving the evaluation history')
        er = EvalRecord(train_ce_frame_sum / b_idx, valid_ce_frame, valid_fer, test_ce_frame, test_fer)
        eval_history.append(er)
        save_eval_history(eval_history, args.save_path + '_eval_history.pkl')

        