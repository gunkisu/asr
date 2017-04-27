#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import pickle
from theano import tensor as T
from collections import namedtuple
import numpy

from lasagne.layers import get_all_params, count_params

from libs.hmrnn_utils import get_arg_parser, get_save_path
from libs.utils import StopWatch, Rsync, gen_win, save_network, save_eval_history, best_fer, show_status, sync_data, \
    find_reload_model, load_or_init_model
from libs.comp_graph_utils import trainer, predictor, eval_net

from libs.lasagne_libs.utils import set_model_param_value
from libs.lasagne_libs.updates import adam
from lasagne.layers import count_params

from libs.deep_lstm_builder import build_deep_lstm
from data.fuel_utils import create_ivector_datastream

from hmrnn.hmlstm_builder import build_graph_am
from hmrnn.mixer import reset_state

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    args.save_path = get_save_path(args)
    print(args)
    
    sw = StopWatch()

    print('Building and compiling the network')
 
    f_prop, f_update, f_log_prob, f_debug, tparams, opt_tparams, \
        states, st_slope = build_graph_am(args)

    sw.print_elapsed()
    sw.reset()

    print('Loading data streams from {}'.format(args.train_dataset))
    sync_data(args)
    datasets = [args.train_dataset, args.valid_dataset, args.test_dataset]
    train_ds, valid_ds, test_ds = [create_ivector_datastream(path=args.data_path, which_set=dataset, 
        batch_size=args.batch_size) for dataset in datasets]

    print('Starting to train')
    epoch_sw = StopWatch()
    status_sw = StopWatch()

    for e_idx in range(1, args.num_epochs+1):
        epoch_sw.reset()
        print('--')
        print('Epoch {} starts'.format(e_idx))
        print('--')
        
        total_ce_sum = 0.0
        total_frame_count = 0
 
        status_sw.reset()
        for b_idx, batch in enumerate(train_ds.get_epoch_iterator(), start=1):
            reset_state(states)
           
            input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = batch

            input_data = numpy.transpose(input_data, (1, 0, 2))
            target_data = numpy.transpose(target_data, (1, 0))
            target_mask = numpy.transpose(target_mask, (1, 0))
            
            cost = f_prop(input_data, target_data, target_mask)
            f_update(args.learn_rate)
            
            print(cost)
            
        print('End of Epoch {}'.format(e_idx))
        epoch_sw.print_elapsed()


        
