#!/usr/bin/env python

from __future__ import print_function

import numpy, theano, lasagne, pickle, os
from theano import tensor as T
from collections import OrderedDict

from libs.deep_lstm_utils import *
from libs.lasagne_libs.updates import momentum
from models.deep_bidir_lstm import deep_bidir_lstm_model

def main(args):
    args.save_path = './wsj_deep_lstm_lr{}_gn{}_gc{}_gs{}_nl{}_nn{}_b{}_iv{}'.format(
            args.learn_rate, args.grad_norm, args.grad_clipping, args.grad_steps, args.num_layers, args.num_nodes, 
            args.batch_size, args.ivector_dim if args.use_ivectors else 0)
    
    reload_path = args.save_path + '_last_model.pkl'

    if os.path.exists(reload_path):
        print('Previously trained model detected: {}'.format(reload_path))
        print('Training will continue with the model')
        args.reload_model = reload_path
    else:
        args.reload_model = None

    if args.use_ivectors:
        args.input_dim = args.input_dim + args.ivector_dim

    print(args)

    print('Build and compile network')
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')
    
    network = deep_bidir_lstm_model(input_var=input_data,
                                    mask_var=input_mask,
                                    num_inputs=args.input_dim,
                                    num_units_list=[args.num_nodes]*args.num_layers,
                                    num_outputs=args.output_dim,
                                    dropout_ratio=args.dropout_ratio,
                                    weight_noise=args.weight_noise,
                                    use_layer_norm=args.use_layer_norm,
                                    peepholes=not args.no_peepholes,
                                    learn_init=args.learn_init,
                                    grad_clipping=args.grad_clipping,
                                    gradient_steps=args.grad_steps,
                                    use_softmax=False)

    network_params = get_all_params(network, trainable=True)

    if args.reload_model:
        print('Loading Parameters...')
        with open(args.reload_model, 'rb') as f:
            pretrain_network_params_val,  pretrain_update_params_val, \
                    pretrain_total_batch_cnt = pickle.load(f)

        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        pretrain_update_params_val = None
        pretrain_total_batch_cnt = 0

    print('Build network trainer')
    training_fn, trainer_params = set_network_trainer(input_data=input_data,
                                                      input_mask=input_mask,
                                                      target_data=target_data,
                                                      target_mask=target_mask,
                                                      num_outputs=args.output_dim,
                                                      network=network,
                                                      updater=eval(args.updater),
                                                      learning_rate=args.learn_rate,
                                                      grad_max_norm=args.grad_norm,
                                                      l2_lambda=args.l2_lambda,
                                                      load_updater_params=pretrain_update_params_val)

    print('Build network predictor')
    predict_fn = set_network_predictor(input_data=input_data,
                                       input_mask=input_mask,
                                       target_data=target_data,
                                       target_mask=target_mask,
                                       num_outputs=args.output_dim,
                                       network=network)


    print('Load data stream')
    train_datastream = get_datastream(path=args.data_path,
                                      which_set=args.train_dataset,
                                      batch_size=args.batch_size, use_ivectors=args.use_ivectors)
    valid_eval_datastream = get_datastream(path=args.data_path,
                                      which_set=args.valid_dataset,
                                      batch_size=args.batch_size, use_ivectors=args.use_ivectors)


    print('Start training')
    evaluation_history =[[[10.0, 10.0, 1.0], [10.0, 10.0 ,1.0]]]
    early_stop_flag = False
    early_stop_cnt = 0
    total_batch_cnt = 0

    try:
        # for each epoch
        for e_idx in range(args.num_epochs):
            # for each batch
            for b_idx, data in enumerate(train_datastream.get_epoch_iterator()):
                total_batch_cnt += 1
                if pretrain_total_batch_cnt>=total_batch_cnt:
                    continue

                # get input, target data
                input_data = data[0].astype(floatX)
                input_mask = data[1].astype(floatX)

                # get target data
                target_data = data[2]
                target_mask = data[3].astype(floatX)

                # get output
                train_output = training_fn(input_data,
                                           input_mask,
                                           target_data,
                                           target_mask)
                train_predict_cost = train_output[0]
                network_grads_norm = train_output[1]

                # show intermediate result
                if total_batch_cnt%args.train_disp_freq== 0 and total_batch_cnt!=0: 
                    show_status(args.save_path, e_idx, total_batch_cnt, train_predict_cost, network_grads_norm, evaluation_history)

#            train_nll, train_bpc, train_fer = eval_net(predict_fn,
#                                                                 train_eval_datastream)
            valid_nll, valid_bpc, valid_fer = eval_net(predict_fn,
                                                                 valid_eval_datastream)

            # check over-fitting
            if valid_fer>evaluation_history[-1][1][2]:
                early_stop_cnt += 1.
            else:
                early_stop_cnt = 0.
                save_network(network_params, trainer_params, total_batch_cnt, args.save_path+'_best_model.pkl') 

            if early_stop_cnt>10:
                print('Training Early Stopped')
                break

            # save results
            evaluation_history.append([[None, None, None],
                                       [valid_nll, valid_bpc, valid_fer]])
            numpy.savez(args.save_path + '_eval_history',
                        eval_history=evaluation_history)

            save_network(network_params, trainer_params, total_batch_cnt, args.save_path + '_last_model.pkl')
 
    except KeyboardInterrupt:
        print('Training Interrupted -- Saving the network and Finishing...')
        save_network(network_params, trainer_params, total_batch_cnt, args.save_path)

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)


