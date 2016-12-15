from argparse import ArgumentParser
import numpy, theano, lasagne, pickle
from theano import tensor as T
from collections import OrderedDict
from data.timit.timit import timit_datastream
from models.baseline import deep_bidir_lstm_model
from lasagne.layers import get_output, get_all_params
from lasagne.regularization import regularize_network_params, l2
from picklable_itertools import groupby
from lasagne.updates import total_norm_constraint
from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values
from libs.lasagne_libs.updates import adamax, nesterov_momentum, adam
from libs.ctc_utils import pseudo_cost as ctc_cost
from libs.ctc_utils import ctc_strip
from libs.eval_utils import Evaluation
from libs.param_utils import set_model_param_value

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

phone_to_phoneme_dict = {'ao':   'aa',
                         'ax':   'ah',
                         'ax-h': 'ah',
                         'axr':  'er',
                         'hv':   'hh',
                         'ix':   'ih',
                         'el':   'l',
                         'em':   'm',
                         'en':   'n',
                         'nx':   'n',
                         'ng':   'eng',
                         'zh':   'sh',
                         'pcl':  'sil',
                         'tcl':  'sil',
                         'kcl':  'sil',
                         'bcl':  'sil',
                         'dcl':  'sil',
                         'gcl':  'sil',
                         'h#':   'sil',
                         'pau':  'sil',
                         'epi':  'sil',
                         'ux':   'uw'}
#################
# BUILD NETWORK #
#################
def build_network(input_data,
                  input_mask,
                  num_inputs=129,
                  num_units_list=(128, 128, 128),
                  num_outputs=64,
                  dropout_ratio=0.2,
                  use_layer_norm=True,
                  weight_noise=0.0,
                  learn_init=True,
                  grad_clipping=1.0):
    # stacked bi-directional lstm (without softmax)
    network = deep_bidir_lstm_model(input_var=input_data,
                                    mask_var=input_mask,
                                    num_inputs=num_inputs,
                                    num_units_list=num_units_list,
                                    num_outputs=num_outputs,
                                    dropout_ratio=dropout_ratio,
                                    use_layer_norm=use_layer_norm,
                                    weight_noise=weight_noise,
                                    learn_init=learn_init,
                                    grad_clipping=grad_clipping,
                                    use_softmax = False)
    return network

#################
# BUILD TRAINER #
#################
def set_network_trainer(input_data,
                        input_mask,
                        target_data,
                        target_mask,
                        network,
                        updater,
                        learning_rate,
                        grad_max_norm=10.,
                        l2_lambda=1e-5,
                        load_updater_params=None):
    ###########################
    # get network output data #
    ###########################
    # get network output data
    network_output = get_output(network, deterministic=False)

    ################################
    # get training cost (CTC + L2) #
    ################################
    # get prediction cost (CTC)
    train_ctc_cost = ctc_cost(y=target_data.dimshuffle(1, 0),
                              y_mask=target_mask.dimshuffle(1, 0),
                              y_hat=network_output.dimshuffle(1, 0, 2),
                              y_hat_mask=input_mask.dimshuffle(1, 0),
                              skip_softmax=True)
    train_ctc_cost = train_ctc_cost.mean()
    # get prediction cost (char-level), CTC)
    train_cost_per_char = train_ctc_cost/target_mask.sum()
    # get regularizer cost (L2)
    train_regularizer_cost = regularize_network_params(network, penalty=l2)

    ##########################
    # get network parameters #
    ##########################
    network_params = get_all_params(network, trainable=True)

    #########################
    # get network gradients #
    #########################
    # get gradient over cost
    network_grads = theano.grad(cost=train_ctc_cost + train_regularizer_cost*l2_lambda,
                                wrt=network_params)
    # get gradient norm constraint
    network_grads, network_grads_norm = total_norm_constraint(tensor_vars=network_grads,
                                                              max_norm=grad_max_norm,
                                                              return_norm=True)

    #######################
    # get network updater #
    #######################
    # get learning rate variable
    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    # get updater
    train_updates, trainer_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    ################################
    # get network updater function #
    ################################
    training_fn = theano.function(inputs=[input_data,
                                          input_mask,
                                          target_data,
                                          target_mask],
                                  outputs=[train_ctc_cost,
                                           train_cost_per_char,
                                           train_regularizer_cost,
                                           network_grads_norm],
                                  updates=train_updates)
    return training_fn, trainer_params

###################
# BUILD PREDICTOR #
###################
def set_network_predictor(input_data,
                          input_mask,
                          target_data,
                          target_mask,
                          network):
    ###########################
    # get network output data #
    ###########################
    network_output = get_output(network, deterministic=True)

    ##################
    # get cost (CTC) #
    ##################
    # get prediction cost (CTC)
    pred_ctc_cost = ctc_cost(y=target_data.dimshuffle(1, 0),
                             y_mask=target_mask.dimshuffle(1, 0),
                             y_hat=network_output.dimshuffle(1, 0, 2),
                             y_hat_mask=input_mask.dimshuffle(1, 0),
                             skip_softmax=True)
    pred_ctc_cost = pred_ctc_cost.mean()
    # get prediction cost (char-level, CTC)
    pred_cost_per_char = pred_ctc_cost/target_mask.sum()
    # get prediction idx
    pred_output_idx = T.argmax(network_output, axis=-1)

    ###########################
    # get prediction function #
    ###########################
    predict_fn = theano.function(inputs=[input_data,
                                         input_mask,
                                         target_data,
                                         target_mask],
                                 outputs=[pred_output_idx,
                                          pred_ctc_cost,
                                          pred_cost_per_char])

    return predict_fn

#############
# EVALUATOR #
#############
def network_evaluation(predict_fn,
                       data_stream,
                       phoneme_dict,
                       black_list):
    #####################
    # get data iterator #
    #####################
    data_iterator = data_stream.get_epoch_iterator()

    #########################
    # set evaluation result #
    #########################
    total_ctc = 0.
    total_per = 0.
    total_sample = 0.
    total_batch = 0.

    ##############
    # evaluation #
    ##############
    # for each batch iteration
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
        predict_ctc_cost = predict_output[1]
        pred_cost_per_char = predict_output[2]

        # add up ctc cost
        total_ctc += predict_ctc_cost
        total_batch += 1.

        # for each data, per evaluation
        for j in range(input_data.shape[0]):
            cur_target_data = target_data[j, :numpy.sum(target_mask[j])]
            cur_predict_data = predict_idx[j, :numpy.sum(input_mask[j])]
            cur_predict_data = ctc_strip(cur_predict_data)

            cur_target_phoneme = [phoneme_dict[phone_ind] for phone_ind in cur_target_data if phoneme_dict[phone_ind] not in black_list]
            cur_predict_phoneme = [phoneme_dict[phone_ind] for phone_ind in cur_predict_data if phoneme_dict[phone_ind] not in black_list]

            targets = [x[0] for x in groupby(cur_target_phoneme)]
            predictions = [x[0] for x in groupby(cur_predict_phoneme)]
            total_per += Evaluation.wer([predictions], [targets])
            total_sample += 1

    total_ctc = total_ctc/total_batch
    total_per = total_per.sum()/total_sample

    return total_ctc, total_per


def main(options):
    #################
    # build network #
    #################
    print 'Build and compile network'
    # input data
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')

    # target data
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')

    # network
    network = build_network(input_data=input_data,
                            input_mask=input_mask,
                            num_inputs=options['num_inputs'],
                            num_units_list=options['num_units_list'],
                            num_outputs=options['num_outputs'],
                            dropout_ratio=options['dropout_ratio'],
                            use_layer_norm=options['use_layer_norm'],
                            weight_noise=options['weight_noise'],
                            learn_init=options['learn_init'],
                            grad_clipping=options['grad_clipping'])
    network_params = get_all_params(network, trainable=True)

    ###################
    # Load Parameters #
    ###################
    if options['reload_model']:
        print('Loading Parameters...')
        pretrain_network_params_val,  pretrain_update_params_val, pretrain_total_batch_cnt = pickle.load(open(options['reload_model'], 'rb'))

        print('Applying Parameters...')
        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        pretrain_update_params_val = None
        pretrain_total_batch_cnt = 0

    #########################
    # build network trainer #
    #########################
    print 'Build network trainer'
    training_fn, trainer_params = set_network_trainer(input_data=input_data,
                                                      input_mask=input_mask,
                                                      target_data=target_data,
                                                      target_mask=target_mask,
                                                      network=network,
                                                      updater=options['updater'],
                                                      learning_rate=options['lr'],
                                                      grad_max_norm=options['grad_norm'],
                                                      l2_lambda=options['l2_lambda'],
                                                      load_updater_params=pretrain_update_params_val)

    ###########################
    # build network predictor #
    ###########################
    print 'Build network predictor'
    predict_fn = set_network_predictor(input_data=input_data,
                                       input_mask=input_mask,
                                       target_data=target_data,
                                       target_mask=target_mask,
                                       network=network)

    ################
    # load dataset #
    ################
    print 'Load data stream'
    train_dataset, train_datastream = timit_datastream(path=options['data_path'],
                                                       which_set='train',
                                                       pool_size=options['pool_size'],
                                                       maximum_frames=options['max_total_frames'],
                                                       local_copy=False)
    valid_dataset, valid_datastream = timit_datastream(path=options['data_path'],
                                                       which_set='dev',
                                                       pool_size=options['pool_size'],
                                                       maximum_frames=options['max_total_frames'],
                                                       local_copy=False)

    phone_dict = train_dataset.get_phoneme_dict()
    phoneme_dict = {k: phone_to_phoneme_dict[v] if v in phone_to_phoneme_dict else v for k, v in phone_dict.iteritems()}
    black_list = ['<START>', '<STOP>', 'q', '<END>']


    ##################
    # start training #
    ##################
    evaluation_history =[[[1000.0, 1.0], [1000.0, 1.0]]]
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
                train_ctc_cost = train_output[0]
                train_cost_per_char = train_output[1]
                train_regularizer_cost = train_output[2]
                network_grads_norm = train_output[3]

                # show intermediate result
                if total_batch_cnt%options['train_disp_freq'] == 0 and total_batch_cnt!=0:
                    print '============================================================================================'
                    print 'Model Name: ', options['save_path'].split('/')[-1]
                    print '============================================================================================'
                    print 'Epoch: ', str(e_idx), ', Update: ', str(total_batch_cnt)
                    print 'CTC Cost: ', str(train_ctc_cost)
                    print 'Per Char Cost: ', str(train_cost_per_char)
                    print 'Regularizer Cost: ', str(train_regularizer_cost)
                    print 'Gradient Norm: ', str(network_grads_norm)
                    print '============================================================================================'
                    print 'Train CTC Cost: ', str(evaluation_history[-1][0][0]), ', PER: ', str(evaluation_history[-1][0][-1])
                    print 'Valid CTC Cost: ', str(evaluation_history[-1][1][0]), ', PER: ', str(evaluation_history[-1][1][-1])

            # evaluation
            train_ctc_cost, train_per = network_evaluation(predict_fn=predict_fn,
                                                           data_stream=train_datastream,
                                                           phoneme_dict=phoneme_dict,
                                                           black_list=black_list)
            valid_ctc_cost, valid_per = network_evaluation(predict_fn=predict_fn,
                                                           data_stream=valid_datastream,
                                                           phoneme_dict=phoneme_dict,
                                                           black_list=black_list)

            # check over-fitting
            if valid_per>evaluation_history[-1][1][-1]:
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
            evaluation_history.append([[train_ctc_cost, train_per],
                                       [valid_ctc_cost, valid_per]])
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
    parser = ArgumentParser()
    # TODO: parser

    options = OrderedDict()
    options['num_units_list'] =  (250, 250, 250, 250, 250)
    options['num_inputs'] = 129
    options['num_outputs'] = 63+1
    options['dropout_ratio'] = 0.2
    options['use_layer_norm'] = True
    options['weight_noise'] = 0.075

    options['updater'] = nesterov_momentum
    options['lr'] = 0.01
    options['grad_norm'] = 10.0
    options['l2_lambda'] = 1e-7
    options['updater_params'] = None
    options['grad_clipping'] = 0.0

    options['pool_size'] = 100
    options['max_total_frames'] = 2000
    options['num_epochs'] = 200

    options['train_disp_freq'] = 10
    options['train_save_freq'] = 100

    options['data_path'] = '/home/kimts/data/speech/timit_alignment.h5'
    options['save_path'] = '/home/kimts/scripts/speech/timit_baseline_ctc'
    options['reload_model'] = None

    options['learn_init'] = True
    options['grad_clipping'] = 1.0

    main(options)







