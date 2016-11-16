from argparse import ArgumentParser
import numpy, theano, lasagne, pickle
from theano import tensor as T
from collections import OrderedDict
from models.baseline import deep_bidir_lstm_model
from lasagne.layers import get_output, get_all_params
from libs.lasagne.utils import get_model_param_values, get_update_params_values
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Padding, FilterSources
from libs.lasagne.updates import momentum
from libs.param_utils import set_model_param_value

import kaldi_io
import sys

floatX = theano.config.floatX

def get_datastream(path, which_set='test_eval92', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    print path, which_set
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    fs = FilterSources(data_stream=base_stream, sources=['features'])
    padded_stream = Padding(data_stream=fs)
    return padded_stream

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
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')

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
        print('Loading model...')
        pretrain_network_params_val,  pretrain_update_params_val, pretrain_total_batch_cnt = pickle.load(open(options['reload_model'], 'rb'))
        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        print 'Must specfiy network to load'
        sys.exit(1)

    ff_fn = ff(input_data=input_data, input_mask=input_mask, network=network)
    test_stream = get_datastream(options['data_path'], options['dataset']) 
    writer = kaldi_io.BaseFloatMatrixWriter(options['save_path'])

    for example in test_stream.get_epoch_iterator():
        input_data, input_mask = example 

        net_output = ff_fn(input_data, input_mask)

        for output in net_output[0]:
            writer.write('uttid', output)

    writer.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO: parser

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

    options['batch_size'] = 12
    options['num_epochs'] = 200

    options['train_disp_freq'] = 10
    options['train_save_freq'] = 100

    options['data_path'] = '/u/songinch/song/data/speech/wsj_fbank123.h5'
    options['dataset'] = 'test_dev93'
    options['save_path'] = 'ark:/u/songinch/song/data/speech/test_dev93_pred.ark'
    options['reload_model'] = '/u/songinch/song/asr/test_model.pkl' 

    main(options)






