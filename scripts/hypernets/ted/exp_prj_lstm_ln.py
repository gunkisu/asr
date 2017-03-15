#!/usr/bin/env python
from __future__ import print_function

import os
import pickle
import argparse
import numpy
import theano
import lasagne
from theano import tensor as T
from collections import namedtuple
from libs.lasagne_libs.utils import set_model_param_value

from lasagne.layers import count_params
from lasagne.layers import get_output, get_all_params
from libs.utils import save_network, save_eval_history, best_fer, show_status, symlink_force
from libs.utils import StopWatch, Rsync
from models.gating_hyper_nets import deep_projection_ln_lstm_model
from data.wsj.fuel_utils import get_datastream
from libs.lasagne_libs.updates import momentum, adam

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

input_dim = 123
output_dim = 4174

def add_params(parser):
    parser.add_argument('--batch_size', default=16, help='batch size', type=int)
    parser.add_argument('--num_layers', default=3, help='number of hidden units', type=int)
    parser.add_argument('--num_units', default=512, help='number of hidden units', type=int)
    parser.add_argument('--learn_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('--grad_clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--dropout', default=0.2, help='dropout', type=float)
    parser.add_argument('--data_path', help='data path', default='/data/lisatmp3/speech/tedlium_fbank123_out4174.h5')
    parser.add_argument('--save_path', help='save path', default='./')
    parser.add_argument('--num_epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--updater', help='sgd or momentum', default='momentum')
    parser.add_argument('--train_disp_freq', help='how ferquently to display progress', default=100, type=int)

    parser.add_argument('--train_dataset', help='dataset for training', default='train')
    parser.add_argument('--valid_dataset', help='dataset for validation', default='dev')
    parser.add_argument('--test_dataset', help='dataset for test', default='test')

    parser.add_argument('--reload_model', help='model path to load')
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally',
                        default='/Tmp/taesup/data/speech')

    parser.add_argument('--no-copy', help='do not copy data from NFS to local machine', action='store_true')

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_params(parser)
    return parser

def get_save_path(args):
    path = args.save_path
    path += '/ted_lstmp_ln'
    path += '_lr{}'.format(args.learn_rate)
    path += '_gc{}'.format(args.grad_clipping)
    path += '_do{}'.format(args.dropout)
    path += '_nl{}'.format(args.num_layers)
    path += '_nu{}'.format(args.num_units)
    path += '_nb{}'.format(args.batch_size)

    return path

def build_trainer(input_data,
                  input_mask,
                  target_data,
                  target_mask,
                  network_params,
                  output_layer,
                  updater,
                  learning_rate,
                  load_updater_params=None):
    output_score = get_output(output_layer, deterministic=False)
    frame_prd_idx = T.argmax(output_score, axis=-1)

    one_hot_target = T.extra_ops.to_one_hot(y=T.flatten(target_data, 1),
                                            nb_class=output_dim,
                                            dtype=floatX)

    output_score = T.reshape(x=output_score,
                             newshape=(-1, output_dim),
                             ndim=2)
    output_score = output_score - T.max(output_score, axis=-1, keepdims=True)
    output_score = output_score - T.log(T.sum(T.exp(output_score), axis=-1, keepdims=True))

    train_ce = -T.sum(T.mul(one_hot_target, output_score), axis=-1)*T.flatten(target_mask, 1)

    train_loss = T.sum(train_ce)/target_mask.shape[0]
    frame_loss = T.sum(train_ce)/T.sum(target_mask)

    frame_accr = T.sum(T.eq(frame_prd_idx, target_data)*target_mask)/T.sum(target_mask)


    train_total_loss = train_loss

    network_grads = theano.grad(cost=train_total_loss, wrt=network_params)
    network_grads_norm = T.sqrt(sum(T.sum(grad**2) for grad in network_grads))

    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_updates, updater_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    training_fn = theano.function(inputs=[input_data,
                                          input_mask,
                                          target_data,
                                          target_mask],
                                  outputs=[frame_loss,
                                           frame_accr,
                                           network_grads_norm],
                                  updates=train_updates)
    return training_fn, train_lr, updater_params

def build_predictor(input_data,
                    input_mask,
                    target_data,
                    target_mask,
                    output_layer):
    output_score = get_output(output_layer, deterministic=True)

    frame_prd_idx = T.argmax(output_score, axis=-1)

    one_hot_target = T.extra_ops.to_one_hot(y=T.flatten(target_data, 1),
                                            nb_class=output_score.shape[-1],
                                            dtype=floatX)

    output_score = T.reshape(x=output_score,
                             newshape=(-1, output_score.shape[-1]),
                             ndim=2)
    output_score = output_score - T.max(output_score, axis=-1, keepdims=True)
    output_score = output_score - T.log(T.sum(T.exp(output_score), axis=-1, keepdims=True))

    frame_loss = -T.sum(T.mul(one_hot_target, output_score), axis=-1)*T.flatten(target_mask, 1)
    frame_loss = T.sum(frame_loss)/T.sum(target_mask)

    return theano.function(inputs=[input_data,
                                   input_mask,
                                   target_data,
                                   target_mask],
                           outputs=[frame_prd_idx,
                                    frame_loss])

def eval_network(predict_fn,
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

if __name__ == '__main__':
    ###############
    # load config #
    ###############
    parser = get_arg_parser()
    args = parser.parse_args()
    args.save_path = get_save_path(args)

    ###################
    # get reload path #
    ###################
    if not args.reload_model:
        reload_path = args.save_path + '_last_model.pkl'

        if os.path.exists(reload_path):
            print('Previously trained model detected: {}'.format(reload_path))
            print('Training continues')
            args.reload_model = reload_path

    ##############
    # print args #
    ##############
    print(args)

    sw = StopWatch()

    if not args.no_copy:
        print('Loading data streams from {}'.format(args.data_path))
        print('Copying data to local machine...')
        rsync = Rsync(args.tmpdir)
        rsync.sync(args.data_path)
        args.data_path = os.path.join(args.tmpdir, os.path.basename(args.data_path))
        sw.print_elapsed()

    ####################
    # load data stream #
    ####################
    train_datastream = get_datastream(path=args.data_path,
                                      which_set=args.train_dataset,
                                      batch_size=args.batch_size)
    valid_datastream = get_datastream(path=args.data_path,
                                      which_set=args.valid_dataset,
                                      batch_size=args.batch_size)
    test_datastream = get_datastream(path=args.data_path,
                                     which_set=args.test_dataset,
                                     batch_size=args.batch_size)

    #################
    # build network #
    #################
    print('Building and compiling network')
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')
    network_output = deep_projection_ln_lstm_model_fix(input_var=input_data,
                                                       mask_var=input_mask,
                                                       num_inputs=input_dim,
                                                       num_outputs=output_dim,
                                                       num_layers=args.num_layers,
                                                       num_units=args.num_units,
                                                       grad_clipping=args.grad_clipping,
                                                       dropout=args.dropout)

    network = network_output
    network_params = get_all_params(network, trainable=True)
    param_count = count_params(network, trainable=True)
    print('Number of parameters of the network: {:.2f}M'.format(float(param_count)/1000000))

    ######################
    # reload model param #
    ######################
    if args.reload_model:
        print('Loading model: {}'.format(args.reload_model))
        with open(args.reload_model, 'rb') as f:
            [pretrain_network_params_val,
             pretrain_update_params_val,
             pretrain_total_batch_cnt] = pickle.load(f)
        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        pretrain_update_params_val = None
        pretrain_total_batch_cnt = 0

    EvalRecord = namedtuple('EvalRecord', ['train_ce_frame',
                                           'valid_ce_frame',
                                           'valid_fer',
                                           'test_ce_frame',
                                           'test_fer'])
    eval_history =[EvalRecord(10.1, 10.0, 1.0, 10.0, 1.0)]


    #################
    # build trainer #
    #################
    print('Building trainer')
    sw.reset()
    [train_fn,
     train_lr,
     updater_params] = build_trainer(input_data=input_data,
                                     input_mask=input_mask,
                                     target_data=target_data,
                                     target_mask=target_mask,
                                     network_params=network_params,
                                     output_layer=network,
                                     updater=eval(args.updater),
                                     learning_rate=args.learn_rate,
                                     load_updater_params=pretrain_update_params_val)
    sw.print_elapsed()

    #################
    # build trainer #
    #################
    print('Building predictor')
    sw.reset()
    predict_fn = build_predictor(input_data=input_data,
                                 input_mask=input_mask,
                                 target_data=target_data,
                                 target_mask=target_mask,
                                 output_layer=network)
    sw.print_elapsed()

    ##################
    # start training #
    ##################
    print('Starting')
    total_batch_cnt = 0
    # for each epoch
    for e_idx in range(1, args.num_epochs+1):
        # control learning rate
        new_lr = args.learn_rate/(1.04**max(e_idx/2-1, 0))
        print("Learning Rate: " + str(new_lr))
        train_lr.set_value(lasagne.utils.floatX(new_lr))

        epoch_sw = StopWatch()
        print('--')
        print('Epoch {} starts'.format(e_idx))
        print('--')

        train_frame_loss_sum = 0.0
        # for each batch
        for batch_idx, batch_data in enumerate(train_datastream.get_epoch_iterator(), start=1):
            total_batch_cnt += 1
            if total_batch_cnt <= pretrain_total_batch_cnt:
                continue

            # get data
            input_data, input_mask, target_data, target_mask = batch_data

            # update model
            train_output = train_fn(input_data,
                                    input_mask,
                                    target_data,
                                    target_mask)
            train_frame_loss = train_output[0]
            train_frame_accr = train_output[1]
            train_grads_norm = train_output[2]
            train_frame_loss_sum += train_frame_loss

            # show results
            if batch_idx%args.train_disp_freq == 0:
                show_status(save_path=args.save_path,
                            ce_frame=train_frame_loss,
                            network_grads_norm=train_grads_norm,
                            batch_idx=batch_idx,
                            batch_size=args.batch_size,
                            epoch_idx=e_idx)
                print('Frame Accr: {}'.format(train_frame_accr))

            if batch_idx % 250 == 0:
                print('Saving the network')
                save_network(network_params=network_params,
                             trainer_params=updater_params,
                             epoch_cnt=total_batch_cnt,
                             save_path=args.save_path + '_last_model.pkl')

            if batch_idx % 1000 == 0:
                print('Evaluating the network on the validation dataset')
                valid_frame_loss, valid_fer = eval_network(predict_fn, valid_datastream)
                test_frame_loss, test_fer = eval_network(predict_fn, test_datastream)
                print('Train CE: {}'.format(train_frame_loss_sum / batch_idx))
                print('Valid CE: {}, FER: {}'.format(valid_frame_loss, valid_fer))
                print('Test  CE: {}, FER: {}'.format(test_frame_loss, test_fer))
                train_frame_loss_sum = 0.0

                if valid_fer < best_fer(eval_history):
                    print('Best model saved based on FER from ' + best_fer(eval_history) + ' to ' + valid_fer)
                    save_network(network_params=network_params,
                                 trainer_params=updater_params,
                                 epoch_cnt=total_batch_cnt,
                                 save_path=args.save_path + '_best_model.pkl')
                print('Saving the evaluation history')
                er = EvalRecord(train_frame_loss_sum / batch_idx, valid_frame_loss, valid_fer, test_frame_loss,
                                test_fer)
                eval_history.append(er)
                save_eval_history(eval_history, args.save_path + '_eval_history.pkl')

        print('End of Epoch {}'.format(e_idx))