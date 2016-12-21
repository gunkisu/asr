#!/usr/bin/env python
from __future__ import print_function
import sys

import numpy, theano, lasagne, pickle, os
from theano import tensor as T
from collections import OrderedDict, namedtuple

from libs.deep_lstm_utils import *
from libs.lasagne_libs.updates import momentum
from libs.utils import StopWatch
from models.deep_bidir_lstm import deep_bidir_lstm_alex

from data.wsj.fuel_utils import get_feat_stream, get_uttid_stream, get_datastream


def main(args):
    args.save_path = './wsj_deep_lstm_lr{}_gn{}_gc{}_gs{}_nl{}_nn{}_b{}_iv{}'.format(
            args.learn_rate, args.grad_norm, args.grad_clipping, args.grad_steps, args.num_layers, args.num_nodes, 
            args.batch_size, args.ivector_dim if args.use_ivectors else 0)
    
    reload_path = args.save_path + '_last_model.pkl'

    if os.path.exists(reload_path):
        print('Previously trained model detected: {}'.format(reload_path))
        print('Training will continue with the model')
        args.reload_model = reload_path
    else:
        args.reload_model = None

    if args.use_ivectors:
        args.input_dim = args.input_dim + args.ivector_dim

    print(args)

    print('Build and compile network')
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')

    
    network = deep_bidir_lstm_alex(input_var=input_data,
                                    mask_var=input_mask,
                                    input_dim=args.input_dim,
                                    num_units_list=[args.num_nodes]*args.num_layers,
                                    output_dim=args.output_dim)

    network_params = get_all_params(network, trainable=True)

    if args.reload_model:
        print('Loading Parameters...')
        with open(args.reload_model, 'rb') as f:
            pretrain_network_params_val,  pretrain_update_params_val, \
                    pretrain_total_epoch_cnt = pickle.load(f)

        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        print('Must provide a pretrained model')
        sys.exit(1)

    print('Build trainer')
    sw = StopWatch()
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

    print('Build predictor')
    sw.reset()
    predict_fn = predictor(input_data=input_data,
                                       input_mask=input_mask,
                                       target_data=target_data,
                                       target_mask=target_mask,
                                       num_outputs=args.output_dim,
                                       network=network)

    sw.print_elapsed()
    print('Load data streams {} and {} from {}'.format(args.train_dataset, args.valid_dataset, args.data_path))
    train_datastream = get_datastream(path=args.data_path,
                                      which_set=args.train_dataset,
                                      batch_size=args.batch_size, use_ivectors=args.use_ivectors)
    valid_eval_datastream = get_datastream(path=args.data_path,
                                      which_set=args.valid_dataset,
                                      batch_size=args.batch_size, use_ivectors=args.use_ivectors)


    print('Start training')
    EvalRecord = namedtuple('EvalRecord', ['ce_frame', 'fer'])

    eval_history =[EvalRecord(10.0, 1.0)]
    early_stop_flag = False
    early_stop_cnt = 0

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
        for b_idx, data in enumerate(train_datastream.get_epoch_iterator(), start=1):
            input_data = data[0].astype(floatX)
            input_mask = data[1].astype(floatX)

            target_data = data[2]
            target_mask = data[3].astype(floatX)

            train_output = training_fn(input_data,
                                       input_mask,
                                       target_data,
                                       target_mask)
            ce_frame = train_output[0]
            network_grads_norm = train_output[1]

            if b_idx%args.train_disp_freq == 0: 
                show_status(args.save_path, ce_frame, network_grads_norm, b_idx, args.batch_size)
            train_ce_frame_sum += ce_frame

        print('End of Epoch {}'.format(e_idx))
        epoch_sw.print_elapsed()
        print('Evaluating the network on the validation dataset')
        eval_sw = StopWatch()
#            train_nll, train_fer = eval_net(predict_fn,
#                                                                 train_eval_datastream)
        valid_ce_frame, valid_fer = eval_net(predict_fn,
        
                                                             valid_eval_datastream)
        eval_sw.print_elapsed()

        if valid_fer>eval_history[-1].fer:
            early_stop_cnt += 1.
        else:
            early_stop_cnt = 0.
            save_network(network_params, trainer_params, e_idx, args.save_path+'_best_model.pkl') 

        print('Train CE: {}'.format(train_ce_frame_sum / b_idx))
        print('Valid CE: {}, FER: {}'.format(valid_ce_frame, valid_fer))

        print('Saving the network and evaluation history')
        eval_history.append(EvalRecord(valid_ce_frame, valid_fer))
        numpy.savez(args.save_path + '_eval_history',
                    eval_history=eval_history)

        save_network(network_params, trainer_params, e_idx, args.save_path + '_last_model.pkl')
        
        if early_stop_cnt>10:
            print('Training early stopped')
            break

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)

