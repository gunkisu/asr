from __future__ import print_function
import argparse
import sys
import numpy, theano, lasagne, pickle, os
from theano import tensor as T
from collections import OrderedDict
from lasagne.layers import get_output, get_all_params
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import total_norm_constraint
from lasagne.objectives import categorical_crossentropy

from six import iteritems
import itertools

from models.deep_bidir_lstm import deep_bidir_lstm_model
from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values
from libs.param_utils import set_model_param_value

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', default=1, help='batch size', type=int)
    parser.add_argument('--num-nodes', default=500, help='number of hidden nodes', type=int)
    parser.add_argument('--num-layers', default=5, help='number of layers', type=int)
    parser.add_argument('--learn-rate', default=0.0001, help='learning rate', type=float)
    parser.add_argument('--grad-norm', default=0.0, help='gradient norm', type=float)
    parser.add_argument('--grad-clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--grad-steps', default=-1, help='gradient steps', type=int)
    parser.add_argument('--use-ivectors', help='whether to use ivectors', action='store_true')
    parser.add_argument('--data-path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--input-dim', help='input dimension', default=123, type=int)
    parser.add_argument('--output-dim', help='output dimension', default=3436, type=int)
    parser.add_argument('--ivector-dim', help='ivector dimension', default=100, type=int)
    parser.add_argument('--no-peepholes', help='do not use peephole connections', action='store_true')
    parser.add_argument('--dropout-ratio', help='dropout ratio', default=0.0, type=float)
    parser.add_argument('--weight-noise', help='weight noise', default=0.0, type=float)
    parser.add_argument('--l2-lambda', help='l2 regularizer', default=0.0, type=float)
    parser.add_argument('--use-layer-norm', help='layer norm', action='store_true')
    parser.add_argument('--learn-init', help='whether to learn initial hidden states', action='store_true')
    parser.add_argument('--num-epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--train-disp-freq', help='how ferquently to display progress', default=100, type=int)
    parser.add_argument('--updater', help='sgd or momentum', default='momentum')
    parser.add_argument('--train-dataset', help='dataset for training', default='train_si84')
    parser.add_argument('--valid-dataset', help='dataset for validation', default='test_dev93')

    return parser

def trainer(input_data,
                        input_mask,
                        target_data,
                        target_mask,
                        num_outputs,
                        network,
                        updater,
                        learning_rate,
                        load_updater_params=None):
    
    o = get_output(network, deterministic=False)
    num_seqs = o.shape[0]
    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))

    ce = ce * T.flatten(target_mask, 1)

    ce_cost = ce.sum()/num_seqs
    ce_frame = ce.sum()/target_mask.sum()

    network_params = get_all_params(network, trainable=True)
    network_grads = theano.grad(cost=ce_cost,
                                wrt=network_params)

    network_grads_norm = T.sqrt(sum(T.sum(grad**2) for grad in network_grads))

    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_updates, trainer_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    training_fn = theano.function(inputs=[input_data,
                                          input_mask,
                                          target_data,
                                          target_mask],
                                  outputs=[ce_frame,
                                           network_grads_norm],
                                  updates=train_updates)
    return training_fn, trainer_params

def predictor(input_data,
                          input_mask,
                          target_data,
                          target_mask,
                          num_outputs,
                          network):
    o = get_output(network, deterministic=False)
    num_seqs = o.shape[0]
    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))

    pred_idx = T.argmax(o, axis=-1)
    ce = ce * T.flatten(target_mask, 1)

    ce_frame = ce.sum()/target_mask.sum()

    return theano.function(inputs=[input_data,
                                         input_mask,
                                         target_data,
                                         target_mask],
                                 outputs=[pred_idx,
                                          ce_frame])

def eval_net(predict_fn,
                       data_stream):

    data_iterator = data_stream.get_epoch_iterator()

    total_nll = 0.
    total_fer = 0.

    for batch_cnt, data in enumerate(data_iterator, start=1):
        input_data = data[0].astype(floatX)
        input_mask = data[1].astype(floatX)

        target_data = data[2]
        target_mask = data[3].astype(floatX)

        predict_output = predict_fn(input_data,
                                    input_mask,
                                    target_data,
                                    target_mask)
        predict_idx = predict_output[0]
        predict_cost = predict_output[1]

        match_data = (target_data == predict_idx)*target_mask
        match_avg = numpy.sum(match_data)/numpy.sum(target_mask)

        total_nll += predict_cost
        total_fer += (1.0 - match_avg)

    total_nll /= batch_cnt 
    total_fer /= batch_cnt

    return total_nll, total_fer


def save_network(network_params, trainer_params, epoch_cnt, save_path):
    cur_network_params_val = get_model_param_values(network_params)
    cur_trainer_params_val = get_update_params_values(trainer_params)
    pickle.dump([cur_network_params_val, cur_trainer_params_val, epoch_cnt],
                open(save_path, 'wb'))


def show_status(save_path, ce_frame, network_grads_norm, batch_idx, batch_size, epoch_idx):
    model = save_path.split('/')[-1]
    print('--')
    print('Model Name: {} (Epoch {})'.format(model, epoch_idx))
    print('Train CE {} (batch {}, {} examples so far): '.format(ce_frame, batch_idx, batch_idx*batch_size))
    print('Gradient Norm: {}'.format(network_grads_norm))
