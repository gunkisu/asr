from __future__ import print_function

import os
import pickle
import sys
from theano import tensor as T
from collections import namedtuple
import numpy

from lasagne.layers import get_all_params, count_params

from libs.deep_lstm_utils import get_arg_parser, get_save_path
from libs.utils import save_network, save_eval_history, best_fer, show_status

from libs.lasagne_libs.utils import set_model_param_value

from models.deep_bidir_lstm import build_deep_bidir_lstm_alex
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
    if args.use_ivector_input:
        ivector_data = T.ftensor3('ivector_data')

    network = build_deep_bidir_lstm_alex(input_var=input_data,
                                    mask_var=input_mask,
                                    input_dim=args.input_dim,
                                    num_units_list=[args.num_nodes]*args.num_layers,
                                    output_dim=args.output_dim, bidir=not args.unidirectional,
                                    ivector_var=ivector_data,
                                    ivector_dim=args.ivector_dim)

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
        if args.use_ivector_input:
            net_output = ff_fn(input_data, input_mask, ivector_data)
        else:
            net_output = ff_fn(input_data, input_mask)

        print('Writing outputs...', file=sys.stderr)
        for out_idx, (output, uttid) in enumerate(zip(net_output[0], uttid_batch[0])):
            valid_len = feat_lens[out_idx]
            writer.write(uttid.encode('ascii'), numpy.log(output[:valid_len]))

    writer.close()
