from argparse import ArgumentParser
import os
import numpy, theano, lasagne, pickle
from theano import tensor as T
from collections import OrderedDict

from models.baseline import deep_bidir_lstm_model
from lasagne.layers import get_output, get_all_params
from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import total_norm_constraint
from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values
from libs.lasagne_libs.updates import adamax, nesterov_momentum, momentum

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Padding, FilterSources

from libs.param_utils import set_model_param_value

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

def get_datastream(path, which_set='train_si84', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    print path, which_set
    iterator_scheme = ShuffledScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    fs = FilterSources(data_stream=base_stream, sources=['features', 'targets'])
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

def set_network_trainer(input_data,
                        input_mask,
                        target_data,
                        target_mask,
                        network,
                        updater,
                        learning_rate,
                        grad_max_norm=10.,
#                        l2_lambda=1e-5,
                        load_updater_params=None):

    # get network output data
    predict_data = get_output(network, deterministic=False)

    predict_idx = T.argmax(predict_data, axis=-1)

    # get prediction cost
    train_predict_cost = categorical_crossentropy(predictions=T.reshape(predict_data, (-1, predict_data.shape[-1])) + eps,
                                                  targets=T.flatten(target_data, 1))
    train_predict_cost = train_predict_cost*T.flatten(target_mask, 1)
    train_predict_cost = train_predict_cost.sum()/target_mask.sum()

    # get regularizer cost
    train_regularizer_cost = regularize_network_params(network, penalty=l2)

    # get network parameters
    network_params = get_all_params(network, trainable=True)

    # get network gradients with clipping
    network_grads = theano.grad(cost=train_predict_cost + train_regularizer_cost*l2_lambda,
                                wrt=network_params)
    network_grads = theano.grad(cost=train_predict_cost,
                                wrt=network_params)
    network_grads, network_grads_norm = total_norm_constraint(tensor_vars=network_grads,
                                                              max_norm=grad_max_norm,
                                                              return_norm=True)


    # set updater
    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_updates, trainer_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    # get training (update) function
    training_fn = theano.function(inputs=[input_data,
                                          input_mask,
                                          target_data,
                                          target_mask],
                                  outputs=[predict_data,
                                           predict_idx,
                                           train_predict_cost,
                                           train_regularizer_cost],
                                           network_grads_norm],
                                  updates=train_updates, allow_input_downcast=True)
    return training_fn, trainer_params

def set_network_predictor(input_data,
                          input_mask,
                          target_data,
                          target_mask,
                          network):

    # get network output data
    predict_data = get_output(network, deterministic=True)

    # get prediction index
    predict_idx = T.argmax(predict_data, axis=-1)

    # get prediction cost
    predict_cost = categorical_crossentropy(predictions=T.reshape(predict_data, (-1, predict_data.shape[-1]))+eps,
                                            targets=T.flatten(target_data, 1))
    predict_cost = predict_cost*T.flatten(target_mask, 1)
    predict_cost = predict_cost.sum()/target_mask.sum()

    # get prediction function
    predict_fn = theano.function(inputs=[input_data,
                                         input_mask,
                                         target_data,
                                         target_mask],
                                 outputs=[predict_idx,
                                          predict_cost], allow_input_downcast=True)

    return predict_fn

def network_evaluation(predict_fn,
                       data_stream):

    data_iterator = data_stream.get_epoch_iterator()

    # evaluation results
    total_nll = 0.
    total_per = 0.
    total_cnt = 0.

    # for each batch
    for i, data in enumerate(data_iterator):
        # get input data
        input_data = data[0]
        input_mask = data[1]

        # get target data
        target_data = data[2]
        target_mask = data[3]

        # get prediction data
        predict_output = predict_fn(input_data,
                                    input_mask,
                                    target_data,
                                    target_mask)
        predict_idx = predict_output[0]
        predict_cost = predict_output[1]

        # compare with target data
        match_data = (target_data == predict_idx)*target_mask

        # average over sequence
        match_avg = numpy.sum(match_data, axis=-1)/numpy.sum(target_mask, axis=-1)
        match_avg = match_avg.mean()

        # add up cost
        total_nll += predict_cost
        total_per += (1.0 - match_avg)
        total_cnt += 1.

    total_nll /= total_cnt
    total_bpc = total_nll/numpy.log(2.0)
    total_per /= total_cnt

    return total_nll, total_bpc, total_per


def main(options):
    print 'Load data stream'

    train_datastream = get_datastream(path=options['data_path'],
                                                  which_set='train_si84',
                                                  batch_size=options['batch_size'])
    valid_datastream = get_datastream(path=options['data_path'],
                                                  which_set='test_dev93',
                                                  batch_size=options['batch_size'])
    print 'Build and compile network'
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
                            learn_init=options['learn_init'],
                            grad_clipping=options['grad_clipping'])
    network_params = get_all_params(network, trainable=True)

    if options['reload_model']:
        print('Loading Parameters...')
        pretrain_network_params_val,  pretrain_update_params_val, pretrain_total_batch_cnt = pickle.load(open(options['reload_model'], 'rb'))

        print('Applying Parameters...')
        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        pretrain_update_params_val = None
        pretrain_total_batch_cnt = 0

    print 'Build network trainer'
    training_fn, trainer_params = set_network_trainer(input_data=input_data,
                                                      input_mask=input_mask,
                                                      target_data=target_data,
                                                      target_mask=target_mask,
                                                      network=network,
                                                      updater=options['updater'],
                                                      learning_rate=options['lr'],
                                                      grad_max_norm=options['grad_max_norm'],
                                                      l2_lambda=options['l2_lambda'],
                                                      load_updater_params=pretrain_update_params_val)

    print 'Build network predictor'
    predict_fn = set_network_predictor(input_data=input_data,
                                       input_mask=input_mask,
                                       target_data=target_data,
                                       target_mask=target_mask,
                                       network=network)


    evaluation_history =[[[10.0, 10.0, 1.0], [10.0, 10.0 ,1.0]]]
    check_early_stop = 0
    total_batch_cnt = 0

    print 'Start training'
    try:
        # for each epoch
        for e_idx in range(options['num_epochs']):
            # for each batch
            for b_idx, data in enumerate(train_datastream.get_epoch_iterator()):
                total_batch_cnt += 1
                if pretrain_total_batch_cnt>=total_batch_cnt:
                    continue
                # get input, target data
                train_input = data

                # get output
                train_output = training_fn(*train_input)
                train_predict_cost = train_output[2]
                train_regularizer_cost = train_output[3]
                network_grads_norm = train_output[3]

                # show intermediate result
                if total_batch_cnt%options['train_disp_freq'] == 0 and total_batch_cnt!=0:
                    print '============================================================================================'
                    print 'Model Name: ', options['save_path'].split('/')[-1]
                    print '============================================================================================'
                    print 'Epoch: ', str(e_idx), ', Update: ', str(total_batch_cnt)
                    print '--------------------------------------------------------------------------------------------'
                    print 'Prediction Cost: ', str(train_predict_cost)
                    print 'Regularizer Cost: ', str(train_regularizer_cost)
                    print 'Gradient Norm: ', str(network_grads_norm)
                    print '--------------------------------------------------------------------------------------------'
                    print 'Train NLL: ', str(evaluation_history[-1][0][0]), ', BPC: ', str(evaluation_history[-1][0][1]), ', FER: ', str(evaluation_history[-1][0][2])
                    print 'Valid NLL: ', str(evaluation_history[-1][1][0]), ', BPC: ', str(evaluation_history[-1][1][1]), ', FER: ', str(evaluation_history[-1][1][2])

            # evaluation
            train_nll, train_bpc, train_per = network_evaluation(predict_fn,
                                                                 train_datastream)
            valid_nll, valid_bpc, valid_per = network_evaluation(predict_fn,
                                                                 valid_datastream)

            # check over-fitting
            if valid_per>evaluation_history[-1][1][2]:
                check_early_stop += 1.
            else:
                check_early_stop = 0.
                best_network_params_vals = get_model_param_values(network_params)
                pickle.dump(best_network_params_vals,
                            open(options['save_path'] + '_best_model.pkl', 'wb'))

            if check_early_stop>10:
                print('Training Early Stopped')
                break

            # save results
            evaluation_history.append([[train_nll, train_bpc, train_per],
                                       [valid_nll, valid_bpc, valid_per]])
            numpy.savez(options['save_path'] + '_eval_history',
                        eval_history=evaluation_history)

            cur_network_params_val = get_model_param_values(network_params)
            cur_trainer_params_val = get_update_params_values(trainer_params)
            cur_total_batch_cnt = total_batch_cnt
            pickle.dump([cur_network_params_val, cur_trainer_params_val, cur_total_batch_cnt],
                        open(options['save_path'] + '_last_model.pkl', 'wb'))

    except KeyboardInterrupt:
        print('Training Interrupted')
        cur_network_params_val = get_model_param_values(network_params)
        cur_trainer_params_val = get_update_params_values(trainer_params)
        cur_total_batch_cnt = total_batch_cnt
        pickle.dump([cur_network_params_val, cur_trainer_params_val, cur_total_batch_cnt],
                    open(options['save_path'] + '_last_model.pkl', 'wb'))

if __name__ == '__main__':
#    parser = ArgumentParser()
#    parser.add_argument('model')
#    args = parser.parse_args()

    options = OrderedDict()
    options['num_units_list'] =  (500, 500, 500, 500, 500)
    options['num_inputs'] = 123
    options['num_outputs'] = 3436
    options['dropout_ratio'] = 0.0
    options['use_layer_norm'] = False
    options['grad_clipping'] = 1.0
    options['learn_init'] = False

    options['updater'] = momentum
    options['lr'] = 0.1
    options['grad_max_norm'] = 10.0
    options['l2_lambda'] = 0
    options['updater_params'] = None

    options['batch_size'] = 2
    options['num_epochs'] = 200

    options['train_disp_freq'] = 10
    options['train_save_freq'] = 100

    options['data_path'] = '/u/songinch/song/data/speech/wsj_fbank123.h5'
    options['save_path'] = '/u/songinch/song/data/exp/wsj_baseline'
    options['reload_model'] = None

    main(options)



