from argparse import ArgumentParser
import argparse
import numpy, theano, lasagne, pickle, os
from theano import tensor as T
from collections import OrderedDict

from deep_lstm_utils import *

def main(options):
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
                            weight_noise=options['weight_noise'],
                            use_layer_norm=options['use_layer_norm'],
                            peepholes=options['peepholes'],
                            learn_init=options['learn_init'],
                            grad_clipping=options['grad_clipping'],
                            gradient_steps=options['gradient_steps'])

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
                                                      num_outputs=options['num_outputs'],
                                                      network=network,
                                                      updater=options['updater'],
                                                      learning_rate=options['lr'],
                                                      grad_max_norm=options['grad_norm'],
                                                      l2_lambda=options['l2_lambda'],
                                                      load_updater_params=pretrain_update_params_val)

    print 'Build network predictor'
    predict_fn = set_network_predictor(input_data=input_data,
                                       input_mask=input_mask,
                                       target_data=target_data,
                                       target_mask=target_mask,
                                       num_outputs=options['num_outputs'],
                                       network=network)


    print 'Load data stream'
    train_datastream = get_datastream(path=options['data_path'],
                                      which_set='train_si84',
                                      batch_size=options['batch_size'], use_ivectors=options['use_ivectors'])
    valid_eval_datastream = get_datastream(path=options['data_path'],
                                      which_set='test_dev93',
                                      batch_size=options['batch_size'], use_ivectors=options['use_ivectors'])


    print 'Start training'
    evaluation_history =[[[10.0, 10.0, 1.0], [10.0, 10.0 ,1.0]]]
    early_stop_flag = False
    early_stop_cnt = 0
    total_batch_cnt = 0

    try:
        # for each epoch
        for e_idx in range(options['num_epochs']):
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
                if total_batch_cnt%options['train_disp_freq'] == 0 and total_batch_cnt!=0: 
                    show_status(options['save_path'], e_idx, total_batch_cnt, train_predict_cost, network_grads_norm, evaluation_history)

#            train_nll, train_bpc, train_fer = eval_net(predict_fn,
#                                                                 train_eval_datastream)
            valid_nll, valid_bpc, valid_fer = eval_net(predict_fn,
                                                                 valid_eval_datastream)

            # check over-fitting
            if valid_fer>evaluation_history[-1][1][2]:
                early_stop_cnt += 1.
            else:
                early_stop_cnt = 0.
                best_network_params_vals = get_model_param_values(network_params)
                pickle.dump(best_network_params_vals,
                            open(options['save_path'] + '_best_model.pkl', 'wb'))

            if early_stop_cnt>10:
                print('Training Early Stopped')
                break

            # save results
            evaluation_history.append([[None, None, None],
                                       [valid_nll, valid_bpc, valid_fer]])
            numpy.savez(options['save_path'] + '_eval_history',
                        eval_history=evaluation_history)

            # save network
            if total_batch_cnt%options['train_save_freq'] == 0 and total_batch_cnt!=0:
                print 'Saving the network'
                save_network(network_params, trainer_params, total_batch_cnt, options['save_path'])
 
    except KeyboardInterrupt:
        print 'Training Interrupted -- Saving the network and Finishing...'
        save_network(network_params, trainer_params, total_batch_cnt, options['save_path'])

if __name__ == '__main__':
    from libs.lasagne.updates import momentum
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', action='store',help='batch size', default=1, type=int)
    parser.add_argument('--num-nodes', action='store',help='num of nodes', default=500, type=int)
    parser.add_argument('--num-layers', action='store',help='num of layers', default=5, type=int)
    parser.add_argument('--learn-rate', action='store', help='learning rate', default=0.0001, type=float)
    parser.add_argument('--grad-norm', action='store', help='gradient norm', default=0.0)
    parser.add_argument('--grad-clipping', action='store', help='gradient clipping', default=1.0)
    parser.add_argument('--grad-steps', action='store', help='gradient steps', default=-1)
    parser.add_argument('--use-ivectors', action='store_true', help='use ivectors')
    parser.add_argument('--data-path', action='store', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')

    args = parser.parse_args()
    grad_norm = float(args.grad_norm)
    grad_clipping = float(args.grad_clipping)
    gradient_steps = int(args.grad_steps)

    ivector_dim = 100
    input_dim = 123

    options = OrderedDict()
    options['use_ivectors'] = args.use_ivectors
    options['num_inputs'] = input_dim+ivector_dim if args.use_ivectors else input_dim
    options['num_units_list'] = [args.num_nodes]*args.num_layers
    options['num_outputs'] = 3436

    options['dropout_ratio'] = 0.0
    options['weight_noise'] = 0.0
    options['use_layer_norm'] = False

    options['peepholes'] = False
    options['learn_init'] = False

    options['updater'] = momentum
    options['lr'] = args.learn_rate
    options['grad_norm'] = grad_norm
    options['grad_clipping'] = grad_clipping
    options['gradient_steps'] = gradient_steps
    #options['l2_lambda'] = 1e-5
    options['l2_lambda'] = 0.0

    options['batch_size'] = args.batch_size
    #options['eval_batch_size'] = 64
    options['num_epochs'] = 200

    options['train_disp_freq'] = 50
    options['train_eval_freq'] = 500
    options['train_save_freq'] = 100

    options['data_path'] = args.data_path

    options['save_path'] = './wsj_deep_lstm_lr{}_gn{}_gc{}_gs{}_nl{}_nn{}_b{}_iv{}'.format(
            args.learn_rate, grad_norm, grad_clipping, gradient_steps, args.num_layers, args.num_nodes, 
            args.batch_size, ivector_dim if args.use_ivectors else 0)

    reload_path = options['save_path'] + '_last_model.pkl'

    if os.path.exists(reload_path):
        options['reload_model'] = reload_path
    else:
        options['reload_model'] = None

    for key, val in options.iteritems():
        print str(key), ': ', str(val)

    main(options)


