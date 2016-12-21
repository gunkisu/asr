from __future__ import print_function

import sys
import numpy, theano, lasagne, pickle
from theano import tensor as T
from lasagne.layers import get_output, get_all_params
from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values
from libs.param_utils import set_model_param_value

from libs.deep_lstm_utils import *
import kaldi_io

from models.deep_bidir_lstm import deep_bidir_lstm_alex
from data.wsj.fuel_utils import get_feat_stream, get_uttid_stream, get_datastream

def ff(input_data, input_mask, network):
    predict_data = get_output(network, deterministic=True)
    predict_fn = theano.function(inputs=[input_data,
                                         input_mask],
                                 outputs=[predict_data])

    return predict_fn

def main(args):
    if args.use_ivectors:
        args.input_dim = args.input_dim + args.ivector_dim
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    
    network = deep_bidir_lstm_alex(input_var=input_data,
                                    mask_var=input_mask,
                                    input_dim=args.input_dim,
                                    num_units_list=[args.num_nodes]*args.num_layers,
                                    output_dim=args.output_dim)

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

    ff_fn = ff(input_data=input_data, input_mask=input_mask, network=network)
    feat_stream = get_feat_stream(args.data_path, args.dataset, args.batch_size, use_ivectors=args.use_ivectors) 
    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.batch_size) 
    
    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    for batch_idx, (feat_batch, uttid_batch) in enumerate(
            zip(feat_stream.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
        input_data, input_mask = feat_batch 
        feat_lens = input_mask.sum(axis=1)

        print('Feed-forwarding...', file=sys.stderr)
        net_output = ff_fn(input_data, input_mask)

        print('Writing outputs...', file=sys.stderr)
        for out_idx, (output, uttid) in enumerate(zip(net_output[0], uttid_batch[0])):
            valid_len = feat_lens[out_idx]
            writer.write(uttid.encode('ascii'), numpy.log(output[:valid_len]))

    writer.close()

if __name__ == '__main__':
    parser = get_arg_parser()
    parser.add_argument('model')
    parser.add_argument('dataset')
    parser.add_argument('wxfilename')

    args = parser.parse_args()
    print(args, file=sys.stderr)

    main(args)
