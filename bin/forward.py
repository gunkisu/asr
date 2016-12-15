from __future__ import print_function

from argparse import ArgumentParser
import numpy, theano, lasagne, pickle
from theano import tensor as T
from collections import OrderedDict
from models.baseline import deep_bidir_lstm_model
from lasagne.layers import get_output, get_all_params
from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Padding, FilterSources
from libs.lasagne_libs.updates import momentum
from libs.param_utils import set_model_param_value

import kaldi_io
import sys

floatX = theano.config.floatX

def get_feat_stream(path, which_set='test_eval92', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    print(path, which_set, file=sys.stderr)
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    fs = FilterSources(data_stream=base_stream, sources=['features'])
    padded_stream = Padding(data_stream=fs)
    return padded_stream

def get_uttid_stream(path, which_set='test_eval92', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    print(path, which_set, file=sys.stderr)
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    fs = FilterSources(data_stream=base_stream, sources=['uttids'])
    return fs

def build_network(input_data,
                  input_mask,
                  num_inputs=123,
                  num_units_list=(128, 128, 128),
                  num_outputs=63,
                  dropout_ratio=0.2,
                  use_layer_norm=True,
                  learn_init=True,
                  grad_clipping=0.0):
    network = deep_bidir_lstm_model(input_var=input_data,
                                    mask_var=input_mask,
                                    num_inputs=num_inputs,
                                    num_units_list=num_units_list,
                                    num_outputs=num_outputs,
                                    dropout_ratio=dropout_ratio,
                                    use_layer_norm=use_layer_norm,
                                    learn_init=learn_init,
                                    grad_clipping=grad_clipping)
    return network

def ff(input_data, input_mask, network):
    predict_data = get_output(network, deterministic=True)
    predict_fn = theano.function(inputs=[input_data,
                                         input_mask],
                                 outputs=[predict_data])

    return predict_fn


def main(options):
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')

    network = build_network(input_data=input_data,
                            input_mask=input_mask,
                            num_inputs=options['num_inputs'],
                            num_units_list=options['num_units_list'],
                            num_outputs=options['num_outputs'],
                            dropout_ratio=options['dropout_ratio'],
                            use_layer_norm=options['use_layer_norm'],
                            learn_init=True,
                            grad_clipping=1.0)
    network_params = get_all_params(network, trainable=True)

    if options['reload_model']:
        print('Loading model...', file=sys.stderr)
        pretrain_network_params_val,  pretrain_update_params_val, pretrain_total_batch_cnt = pickle.load(open(options['reload_model'], 'rb'))
        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        print('Must specfiy network to load', file=sys.stderr)
        sys.exit(1)

    ff_fn = ff(input_data=input_data, input_mask=input_mask, network=network)
    feat_stream = get_feat_stream(options['data_path'], options['dataset'], options['batch_size']) 
    uttid_stream = get_uttid_stream(options['data_path'], options['dataset'], options['batch_size']) 
    
    writer = kaldi_io.BaseFloatMatrixWriter(options['save_path'])

    for batch_idx, (feat_batch, uttid_batch) in enumerate(zip(feat_stream.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
        input_data, input_mask = feat_batch 
        feat_lens = input_mask.sum(axis=1)

#        import ipdb; ipdb.set_trace()

        print('Feed-forwarding...', file=sys.stderr)
        net_output = ff_fn(input_data, input_mask)

        print('Writing outputs...', file=sys.stderr)

        for out_idx, (output, uttid) in enumerate(zip(net_output[0], uttid_batch[0])):
            # write log probabilities
            valid_len = feat_lens[out_idx]
            writer.write(uttid.encode('ascii'), numpy.log(output[:valid_len]))

    writer.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('dataset')
    parser.add_argument('wxfilename')

    args = parser.parse_args()

    options = OrderedDict()
    options['num_units_list'] =  (100, 100)
    options['num_inputs'] = 123
    options['num_outputs'] = 3436
    options['dropout_ratio'] = 0.0
    options['use_layer_norm'] = False

    options['updater'] = momentum
    options['lr'] = 0.1
    options['grad_norm'] = 10.0
    options['l2_lambda'] = 0
    options['updater_params'] = None

    options['batch_size'] = 8 
    options['num_epochs'] = 200

    options['train_disp_freq'] = 10
    options['train_save_freq'] = 100

    options['data_path'] = '/u/songinch/song/data/speech/wsj_fbank123.h5'
    options['dataset'] = args.dataset
    options['save_path'] = args.wxfilename 
    options['reload_model'] = args.model

    main(options)
