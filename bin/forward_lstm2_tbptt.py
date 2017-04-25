#!/usr/bin/env python
from __future__ import print_function

import os
import pickle
import sys
from theano import tensor as T
from collections import namedtuple
import numpy

from lasagne.layers import get_all_params, count_params

from libs.deep_lstm_utils import get_arg_parser, get_save_path
from libs.utils import save_network, save_eval_history, best_fer, show_status, \
    skip_frames, gen_win_test

from libs.lasagne_libs.utils import set_model_param_value

from libs.deep_lstm_builder import build_deep_lstm_tbptt
from data.fuel_utils import create_ivector_test_datastream, get_uttid_stream

from libs.comp_graph_utils import ff_tbptt

import kaldi_io

if __name__ == '__main__':
    parser = get_arg_parser()

    parser.add_argument('model')
    parser.add_argument('dataset')
    parser.add_argument('wxfilename')

    args = parser.parse_args()


    print(args, file=sys.stderr)

    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    
    ivector_data = None
    if args.use_ivector_input:
        ivector_data = T.ftensor3('ivector_data')
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
                                context=args.num_tbptt_steps,
                                grad_clipping=args.grad_clipping,
                                is_bidir=not args.uni,
                                use_layer_norm=args.use_layer_norm,
                                ivector_dim=args.ivector_dim,
                                ivector_var=ivector_data, backward_on_top=args.backward_on_top)

    network_params = get_all_params(network, trainable=True)

    print('Loading Parameters...', file=sys.stderr)
    if args.model:
        with open(args.model, 'rb') as f:
            pretrain_network_params_val,  pretrain_update_params_val, \
                    pretrain_total_epoch_cnt = pickle.load(f)

            set_model_param_value(network_params, pretrain_network_params_val)
    else:
        print('Must specfiy network to load', file=sys.stderr)
        sys.exit(1)

    ff_fn = ff_tbptt(input_data, input_mask, network, is_first_win, args.delay, args.right_context, ivector_data)
    test_ds = create_ivector_test_datastream(args.data_path, args.dataset, args.batch_size)
    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.batch_size) 

    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    for batch_idx, (feat_batch, uttid_batch) in enumerate(
            zip(test_ds.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
        i_data, i_mask, _, _ = feat_batch
        n_batch, _, _ = i_data.shape
        feat_lens = i_mask.sum(axis=1)

        uttid_batch, = uttid_batch

        for l in tbptt_layers:
            l.reset(n_batch)
        
        print('Feed-forwarding...', file=sys.stderr)

        net_output_list = []
        for win_idx, win in enumerate(gen_win_test(feat_batch, args.num_tbptt_steps, args.right_context), start=1):

            input_data, input_mask, ivector_data, ivector_mask = win
            is_first_win = 1 if win_idx == 1 else 0

            if args.use_ivector_input:
                net_output, = ff_fn(input_data, input_mask, ivector_data, is_first_win)
            else:
                net_output, = ff_fn(input_data, input_mask, is_first_win)
        
            net_output_list.append(net_output)

        net_output = numpy.concatenate(net_output_list, axis=1)
        
        print('Writing outputs...', file=sys.stderr)
        for out_idx, (output, uttid) in enumerate(zip(net_output, uttid_batch)):
            valid_len = feat_lens[out_idx]
            writer.write(uttid.encode('ascii'), numpy.log(output[:valid_len]))

    writer.close()
