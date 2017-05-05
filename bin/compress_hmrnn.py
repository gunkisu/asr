#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import pickle
from theano import tensor as T
from collections import namedtuple, OrderedDict
import numpy

from lasagne.layers import get_all_params, count_params

from libs.hmrnn_utils import get_arg_parser, get_save_path
from libs.utils import StopWatch, Rsync, gen_win, save_network, save_eval_history, best_fer, show_status2, sync_data, \
    find_reload_model, load_or_init_model, compress_batch
from libs.comp_graph_utils import trainer, predictor, eval_net

from libs.lasagne_libs.utils import set_model_param_value
from libs.lasagne_libs.updates import adam
from lasagne.layers import count_params

from libs.deep_lstm_builder import build_deep_lstm
from data.fuel_utils import create_ivector_datastream, create_ivector_test_datastream

from hmrnn.hmlstm_builder import build_graph_am
from hmrnn.mixer import reset_state, insert_item2dict, unzip, save_npz, save_npz2, init_tparams_with_restored_value

def eval_model(ds, states, f_log_prob):
    total_ce_sum = 0.0
    total_frame_count = 0

    for b_idx, batch in enumerate(ds.get_epoch_iterator(), start=1):
        reset_state(states)
       
        input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = batch
        n_batch, n_seq, n_feat = input_data.shape
        if n_batch < args.batch_size:
            continue

        input_data = numpy.transpose(input_data, (1, 0, 2))
        target_data = numpy.transpose(target_data, (1, 0))
        target_mask = numpy.transpose(target_mask, (1, 0))
        
        cost, cost_len = f_log_prob(input_data, target_data, target_mask)
        total_ce_sum += cost.sum()
        total_frame_count += cost_len.sum()

    return total_ce_sum / total_frame_count

def batch_mean(batch): 
    mean_sum = 0.0

    for b in batch:
        mean_sum += b.mean()
    
    return mean_sum / len(batch)


def avg_z_1_3d(ds, states, f_debug):
    total_ce_sum = 0.0
    total_frame_count = 0

    z_1_3d_list = []

    for b_idx, batch in enumerate(ds.get_epoch_iterator(), start=1):
        reset_state(states)
       
        input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = batch
        n_batch, n_seq, n_feat = input_data.shape
        if n_batch < args.batch_size:
            continue

        input_data = numpy.transpose(input_data, (1, 0, 2))
        target_data = numpy.transpose(target_data, (1, 0))
        target_mask = numpy.transpose(target_mask, (1, 0))

        h_rnn_1_3d, h_rnn_2_3d, h_rnn_3_3d, z_1_3d, z_2_3d = \
            f_debug(input_data, target_data, target_mask)

        z_1_3d_list.append(z_1_3d)
        
    return batch_mean(z_1_3d_list)


if __name__ == '__main__':
    print(' '.join(sys.argv))

    parser = get_arg_parser()
    parser.add_argument('model')
    parser.add_argument('dataset')
#    parser.add_argument('wxfilename')
    
    args = parser.parse_args()
    print(args)

    sw = StopWatch()
 
    f_prop, f_update, f_log_prob, f_debug, tparams, opt_tparams, \
        states, st_slope = build_graph_am(args)

    sw.print_elapsed()
    sw.reset()

    if not os.path.exists(args.model):
        print('File not found: {}'.format(args.model))
        sys.exit(1)

    tparams = init_tparams_with_restored_value(tparams, args.model)
 
    print('Loading data streams from {}'.format(args.data_path))
    sync_data(args)

    ds = create_ivector_test_datastream(args.data_path, args.dataset, args.batch_size)

    print('Starting to train')
    epoch_sw = StopWatch()
    status_sw = StopWatch()
    
    status_sw.reset()
    z_1_3d_list = []

    for b_idx, batch in enumerate(ds.get_epoch_iterator(), start=1):
        reset_state(states)
       
        input_data, input_mask, ivector_data, ivector_mask = batch
        n_batch, n_seq, n_feat = input_data.shape
        if n_batch < args.batch_size:
            continue

        input_data_trans = numpy.transpose(input_data, (1, 0, 2))
        _, _, _, z_1_3d, _ = f_debug(input_data_trans)

        z_1_3d_trans = numpy.transpose(z_1_3d, (1,0))
        compressed_input_data = compress_batch(input_data, z_1_3d_trans)
        compressed_input_mask = compress_batch(input_mask, z_1_3d_trans)

        
        


