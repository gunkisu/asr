from __future__ import print_function

import pickle
import sys
import argparse
import theano
from theano import tensor as T
import numpy

from lasagne.layers import get_all_params, count_params
from lasagne.layers import get_output
from libs.lasagne_libs.utils import set_model_param_value
from models.gating_hyper_nets import deep_projection_ln_lstm_model_fix
from data.wsj.fuel_utils import get_feat_stream, get_uttid_stream

import kaldi_io

floatX = theano.config.floatX
input_dim = 123
output_dim = 3436

def add_params(parser):
    parser.add_argument('--batch_size', default=16, help='batch size', type=int)
    parser.add_argument('--num_layers', default=3, help='number of hidden units', type=int)
    parser.add_argument('--num_units', default=512, help='number of hidden units', type=int)
    parser.add_argument('--learn_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('--grad_clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--dropout', default=0.2, help='dropout', type=float)
    parser.add_argument('--data_path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--save_path', help='save path', default='./')
    parser.add_argument('--num_epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--updater', help='sgd or momentum', default='momentum')
    parser.add_argument('--train_disp_freq', help='how ferquently to display progress', default=100, type=int)

    parser.add_argument('--train_dataset', help='dataset for training', default='train_si284')
    parser.add_argument('--valid_dataset', help='dataset for validation', default='test_dev93')
    parser.add_argument('--test_dataset', help='dataset for test', default='test_eval92')

    parser.add_argument('--reload_model', help='model path to load')
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally',
                        default='/Tmp/taesup/data/speech')

    parser.add_argument('--no-copy', help='do not copy data from NFS to local machine', action='store_true')

    parser.add_argument('--model', default=None )
    parser.add_argument('--dataset', default='test_eval92')
    parser.add_argument('wxfilename')

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_params(parser)
    return parser


def ff(network, input_data, input_mask):
    predict_data = get_output(network, deterministic=True)
    predict_data = predict_data - T.max(predict_data, axis=-1, keepdims=True)
    predict_data = predict_data - T.log(T.sum(T.exp(predict_data), axis=-1, keepdims=True))
    inputs = [input_data, input_mask]
    predict_fn = theano.function(inputs=inputs,
                                 outputs=[predict_data])
    return predict_fn

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    print(args, file=sys.stderr)

    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')

    network = deep_projection_ln_lstm_model_fix(input_var=input_data,
                                                mask_var=input_mask,
                                                num_inputs=input_dim,
                                                num_outputs=output_dim,
                                                num_layers=args.num_layers,
                                                num_units=args.num_units,
                                                grad_clipping=args.grad_clipping,
                                                dropout=args.dropout)

    network_params = get_all_params(network, trainable=True)
    param_count = count_params(network, trainable=True)
    print('Number of parameters of the network: {:.2f}M'.format(float(param_count) / 1000000), file=sys.stderr)

    print('Loading Parameters...', file=sys.stderr)
    if args.model:
        with open(args.model, 'rb') as f:
            [pretrain_network_params_val,
             pretrain_update_params_val,
             pretrain_total_epoch_cnt] = pickle.load(f)
        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        print('Must specfiy network to load', file=sys.stderr)
        sys.exit(1)

    ff_fn = ff(network, input_data, input_mask)
    test_datastream = get_feat_stream(path=args.data_path,
                                      which_set=args.dataset,
                                      batch_size=args.batch_size)
    uttid_datastream = get_uttid_stream(path=args.data_path,
                                        which_set=args.dataset,
                                        batch_size=args.batch_size)

    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    for batch_idx, (feat_batch, uttid_batch) in enumerate(zip(test_datastream.get_epoch_iterator(),
                                                              uttid_datastream.get_epoch_iterator())):
        input_data, input_mask = feat_batch
        feat_lens = input_mask.sum(axis=1)

        print('Feed-forwarding...', file=sys.stderr)
        net_output = ff_fn(input_data, input_mask)

        print('Writing outputs...', file=sys.stderr)
        for out_idx, (output, uttid) in enumerate(zip(net_output[0], uttid_batch[0])):
            valid_len = int(feat_lens[out_idx])
            writer.write(uttid.encode('ascii'), output[:valid_len])

    writer.close()
