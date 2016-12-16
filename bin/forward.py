from __future__ import print_function
import sys
import numpy, theano, lasagne, pickle
from theano import tensor as T
from collections import OrderedDict
from lasagne.layers import get_output, get_all_params
from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values
from libs.param_utils import set_model_param_value
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Padding, FilterSources

from libs.lasagne_libs.updates import momentum

from libs.deep_lstm_utils import *
import kaldi_io

def ff(input_data, input_mask, network):
    predict_data = get_output(network, deterministic=True)
    predict_fn = theano.function(inputs=[input_data,
                                         input_mask],
                                 outputs=[predict_data])

    return predict_fn

def main(args):
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')

    network = build_network(input_data=input_data,
                            input_mask=input_mask,
                            num_inputs=args.input_dim,
                            num_units_list=[args.num_nodes]*args.num_layers,
                            num_outputs=args.output_dim,
                            dropout_ratio=args.dropout_ratio,
                            weight_noise=args.weight_noise,
                            use_layer_norm=args.use_layer_norm,
                            peepholes=not args.no_peepholes,
                            learn_init=args.learn_init,
                            grad_clipping=args.grad_clipping,
                            gradient_steps=args.grad_steps)
    network_params = get_all_params(network, trainable=True)

    print('Loading Parameters...')
    if args.model:
        with open(args.model, 'rb') as f:
            pretrain_network_params_val,  pretrain_update_params_val, \
                    pretrain_total_batch_cnt = pickle.load(f)

            set_model_param_value(network_params, pretrain_network_params_val)
    else:
        print('Must specfiy network to load')
        sys.exit(1)

    ff_fn = ff(input_data=input_data, input_mask=input_mask, network=network)
    feat_stream = get_feat_stream(args.data_path, args.dataset, args.batch_size) 
    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.batch_size) 
    
    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    for batch_idx, (feat_batch, uttid_batch) in enumerate(
            zip(feat_stream.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
        input_data, input_mask = feat_batch 
        feat_lens = input_mask.sum(axis=1)

        print('Feed-forwarding...')
        net_output = ff_fn(input_data, input_mask)

        print('Writing outputs...')
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
    print(args)

    main(args)
