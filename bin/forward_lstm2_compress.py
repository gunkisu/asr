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
from libs.utils import save_network, save_eval_history, best_fer, show_status, skip_frames, \
        compress_batch, seg_len_info, uncompress_batch

from libs.lasagne_libs.utils import set_model_param_value

from libs.deep_lstm_builder import build_deep_lstm
from data.wsj.fuel_utils import create_ivector_test_datastream, get_uttid_stream

from libs.comp_graph_utils import ff

from hmrnn.hmlstm_builder import build_graph_am
from hmrnn.mixer import reset_state, init_tparams_with_restored_value

from libs.hmrnn_utils import add_hmrnn_graph_params

import kaldi_io

if __name__ == '__main__':
    parser = get_arg_parser()
    add_hmrnn_graph_params(parser)

    parser.add_argument('model')
    parser.add_argument('hmrnn_model')
    parser.add_argument('dataset')
    parser.add_argument('wxfilename')

    args = parser.parse_args()

    print(args, file=sys.stderr)

    
    if args.batch_size != args.n_batch:
        print('--batch-size != --n-batch')
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print('File not found: {}'.format(args.model))
        sys.exit(1)

    if not os.path.exists(args.hmrnn_model):
        print('File not found: {}'.format(args.hmrnn_model))
        sys.exit(1)

    print('Loading an hmrnn model')
    f_prop, f_update, f_log_prob, f_debug, tparams, opt_tparams, \
        states, st_slope = build_graph_am(args)
  
    tparams = init_tparams_with_restored_value(tparams, args.hmrnn_model)

    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    
    ivector_data = None
    if args.use_ivector_input:
        ivector_data = T.ftensor3('ivector_data')

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

    print('Loading Parameters...', file=sys.stderr)
    with open(args.model, 'rb') as f:
        pretrain_network_params_val,  pretrain_update_params_val, \
                pretrain_total_epoch_cnt = pickle.load(f)

        set_model_param_value(network_params, pretrain_network_params_val)

    ff_fn = ff(network, input_data, input_mask, ivector_data)
    test_ds = create_ivector_test_datastream(args.data_path, args.dataset, args.batch_size)
    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.batch_size) 

    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    for batch_idx, (feat_batch, uttid_batch) in enumerate(
            zip(test_ds.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
        input_data, input_mask, ivector_data, ivector_mask = feat_batch 

        n_batch, n_seq, n_feat = input_data.shape
        if n_batch != args.batch_size:
            continue
                    
        uttid_batch, = uttid_batch
        feat_lens = input_mask.sum(axis=1)

        print('Feed-forwarding...', file=sys.stderr)

        input_data_trans = numpy.transpose(input_data, (1, 0, 2))
        _, _, _, z_1_3d, _ = f_debug(input_data_trans)

        z_1_3d_trans = numpy.transpose(z_1_3d, (1,0))
        compressed_batch = [compress_batch(src, z_1_3d_trans) for src in feat_batch]
        len_info = seg_len_info(z_1_3d_trans)
        
        if args.use_ivector_input:
            net_output, = ff_fn(*compressed_batch[:3])
        else:
            net_output, = ff_fn(*compressed_batch[:2])

        net_output = uncompress_batch(net_output, len_info)

        print('Writing outputs...', file=sys.stderr)
        for out_idx, (output, uttid) in enumerate(zip(net_output, uttid_batch)):
            valid_len = feat_lens[out_idx]
            writer.write(uttid.encode('ascii'), numpy.log(output[:valid_len]))

    writer.close()
