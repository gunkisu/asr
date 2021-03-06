from __future__ import print_function

import os
import pickle
import sys
from theano import tensor as T
from collections import namedtuple
import numpy

from lasagne.layers import get_all_params, count_params
from libs.hyperlstm_utils import get_arg_parser, get_save_path

from libs.lasagne_libs.utils import set_model_param_value

from models.deep_hyperlstm import build_deep_hyper_lstm
from data.wsj.fuel_utils import create_ivector_test_datastream, get_uttid_stream

from libs.comp_graph_utils import ff

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
    if args.use_ivector_input or args.use_ivector_model:
        ivector_data = T.ftensor3('ivector_data')

    if args.layer_name == 'IVectorLHUCLSTMLayer' and not args.use_ivector_model:
        print('--use-ivector-model not specified for IVectorLHUCLSTMLayer')
        sys.exit(1)

    if args.layer_name == 'SummarizingLHUCLSTMLayer' and args.use_ivector_model:
        print('--use-ivector-model specified for SummarizingLHUCLSTMLayer')
        sys.exit(1)

    network = build_deep_hyper_lstm(args.layer_name, input_var=input_data,
                             mask_var=input_mask,
                             input_dim=args.input_dim,
                             num_layers=args.num_layers,
                             num_units=args.num_nodes,
                             num_hyper_units=args.num_hyper_nodes,
                             num_proj_units=args.num_proj_nodes,
                             output_dim=args.output_dim,
                             grad_clipping=args.grad_clipping,
                             bidir=not args.unidirectional, 
                             num_hyperlstm_layers=args.num_hyperlstm_layers,
                             use_ivector_input=args.use_ivector_input,
                             ivector_var=ivector_data, 
                             ivector_dim=args.ivector_dim, 
                             reparam=args.reparam, 
                             use_layer_norm=args.use_layer_norm,
                             num_pred_layers=args.num_pred_layers,
                             num_pred_units=args.num_pred_nodes,
                             pred_act=args.pred_act, 
                             num_seqsum_nodes=args.num_seqsum_nodes, 
                             num_seqsum_layers=args.num_seqsum_layers, 
                             seqsum_output_dim=args.seqsum_output_dim)


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

    ff_fn = ff(network, input_data, input_mask, ivector_data)
    test_ds = create_ivector_test_datastream(args.data_path, args.dataset, args.batch_size)
    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.batch_size) 

    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    for batch_idx, (feat_batch, uttid_batch) in enumerate(
            zip(test_ds.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
        input_data, input_mask, ivector_data, ivector_mask = feat_batch 
        feat_lens = input_mask.sum(axis=1)

        print('Feed-forwarding...', file=sys.stderr)
        if args.use_ivector_input or args.use_ivector_model:
            net_output = ff_fn(input_data, input_mask, ivector_data)
        else:
            net_output = ff_fn(input_data, input_mask)

        print('Writing outputs...', file=sys.stderr)
        for out_idx, (output, uttid) in enumerate(zip(net_output[0], uttid_batch[0])):
            valid_len = feat_lens[out_idx]
            writer.write(uttid.encode('ascii'), numpy.log(output[:valid_len]))

    writer.close()
