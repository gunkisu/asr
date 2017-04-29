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
    find_reload_model, load_or_init_model
from libs.comp_graph_utils import trainer, predictor, eval_net

from libs.lasagne_libs.utils import set_model_param_value
from libs.lasagne_libs.updates import adam
from lasagne.layers import count_params

from libs.deep_lstm_builder import build_deep_lstm
from data.fuel_utils import create_ivector_datastream

from hmrnn.hmlstm_builder import build_graph_am
from hmrnn.mixer import reset_state, insert_item2dict, unzip, save_npz

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
    parser = get_arg_parser()
    args = parser.parse_args()

    args.save_path = os.path.join(args.log_dir, get_save_path(args))
    
    print(args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    file_name = '{}.npz'.format(args.save_path)
    best_file_name = '{}.best.npz'.format(args.save_path)
    opt_file_name = '{}.grads.npz'.format(args.save_path)
    best_opt_file_name = '{}.best.grads.npz'.format(args.save_path)
    
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

    summary = OrderedDict()
    _best_score = numpy.iinfo(numpy.int32).max

    global_step = 0
    epoch_step = 0
    batch_step = 0

    for e_idx in range(1, args.num_epochs+1):
        epoch_sw.reset()
        print('--')
        print('Epoch {} starts'.format(e_idx))
        print('--')
        
        total_ce_sum = 0.0
        total_frame_count = 0
 
        status_sw.reset()
        batch_step = 0
        z_1_3d_list = []
        for b_idx, batch in enumerate(train_ds.get_epoch_iterator(), start=1):
            reset_state(states)
           
            input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = batch
            n_batch, n_seq, n_feat = input_data.shape
            if n_batch < args.batch_size:
                continue

            input_data = numpy.transpose(input_data, (1, 0, 2))
            target_data = numpy.transpose(target_data, (1, 0))
            target_mask = numpy.transpose(target_mask, (1, 0))
            
            cost, = f_prop(input_data, target_data, target_mask)
            tr_cost, tr_cost_len = f_log_prob(input_data, target_data, target_mask)
            total_ce_sum += tr_cost.sum()
            total_frame_count += tr_cost_len.sum()

            _, _, _, z_1_3d, _ = f_debug(input_data, target_data, target_mask)
            z_1_3d_list.append(z_1_3d)
            
            f_update(args.learn_rate)
            
            if b_idx%args.log_freq == 0: 

                show_status2(args.save_path, total_ce_sum / total_frame_count, b_idx, args.batch_size, e_idx, batch_mean(z_1_3d_list))
                z_1_3d_list = []
                status_sw.print_elapsed(); status_sw.reset()

            batch_step += 1; global_step += 1
        
        print('End of Epoch {}'.format(e_idx))
        epoch_sw.print_elapsed()
        
        insert_item2dict(summary, 'time', epoch_sw.elapsed())
                
        val_nats = eval_model(valid_ds, states, f_log_prob)
        insert_item2dict(summary, 'val_nats', val_nats)
        val_avg_z_1_3d = avg_z_1_3d(valid_ds, states, f_debug)
        insert_item2dict(summary, 'avg_z_1_3d', val_avg_z_1_3d)
        
        test_nats = eval_model(test_ds, states, f_log_prob)
        insert_item2dict(summary, 'test_nats', test_nats)
        test_avg_z_1_3d = avg_z_1_3d(test_ds, states, f_debug)
        insert_item2dict(summary, 'test_z_1_3d', test_avg_z_1_3d)
        
        if val_nats < _best_score:
            _best_score = val_nats

            best_params = unzip(tparams)
            save_npz(best_file_name, global_step, epoch_step, batch_step,
                 best_params, summary)
            best_opt_params = unzip(opt_tparams)
            save_npz2(best_opt_file_name, best_opt_params)
            print("Best checkpoint stored in: %s" % best_file_name)

        params = unzip(tparams)
        save_npz(file_name, global_step, epoch_step, batch_step, params, summary)
        opt_params = unzip(opt_tparams)
        save_npz2(opt_file_name, opt_params)
        print("Checkpointed in: %s" % file_name)

        epoch_step += 1

        
