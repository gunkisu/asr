from argparse import ArgumentParser
import numpy, theano, lasagne, pickle, os
from theano import tensor as T
from collections import OrderedDict
from models.skip_nets import deep_bidir_diff_skip_lstm_model
from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values
from libs.param_utils import set_model_param_value
from lasagne.layers import get_output, get_all_params, count_params
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import total_norm_constraint
from lasagne.random import set_rng
from lasagne.utils import floatX as convert_to_floatX

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Padding, FilterSources
from data.transformers import Normalize

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

set_rng(numpy.random.RandomState(111))

def get_datastream(path, norm_path, which_set='train_si84', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    data_mean_std = numpy.load(norm_path)

    iterator_scheme = ShuffledScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    base_stream = Normalize(data_stream=base_stream, means=data_mean_std['mean'], stds=data_mean_std['std'])
    fs = FilterSources(data_stream=base_stream, sources=['features', 'targets'])
    padded_stream = Padding(data_stream=fs)
    return padded_stream

def build_network(input_data,
                  input_mask,
                  num_inputs,
                  num_units_list,
                  num_outputs,
                  skip_scale,
                  stochastic,
                  dropout_ratio=0.2,
                  weight_noise=0.0,
                  use_layer_norm=True,
                  peepholes=False,
                  learn_init=True,
                  grad_clipping=0.0,
                  gradient_steps=-1,
                  use_projection=False):

    network, rand_layer_list = deep_bidir_diff_skip_lstm_model(input_var=input_data,
                                                               mask_var=input_mask,
                                                               num_inputs=num_inputs,
                                                               num_units_list=num_units_list,
                                                               num_outputs=num_outputs,
                                                               stochastic=stochastic,
                                                               skip_scale=skip_scale,
                                                               dropout_ratio=dropout_ratio,
                                                               weight_noise=weight_noise,
                                                               use_layer_norm=use_layer_norm,
                                                               peepholes=peepholes,
                                                               learn_init=learn_init,
                                                               grad_clipping=grad_clipping,
                                                               gradient_steps=gradient_steps,
                                                               use_softmax=False,
                                                               use_projection=use_projection)
    return network, rand_layer_list

def set_network_trainer(input_data,
                        input_mask,
                        target_data,
                        target_mask,
                        num_outputs,
                        network,
                        rand_layer_list,
                        updater,
                        learning_rate,
                        grad_max_norm=10.,
                        l2_lambda=1e-5,
                        load_updater_params=None):
    # get one hot target
    one_hot_target_data = T.extra_ops.to_one_hot(y=T.flatten(target_data, 1),
                                                 nb_class=num_outputs,
                                                 dtype=floatX)

    # get network output data
    predict_data = get_output(network, deterministic=False)
    num_seqs = predict_data.shape[0]

    # get prediction cost
    predict_data = T.reshape(x=predict_data,
                             newshape=(-1, num_outputs),
                             ndim=2)
    predict_data = predict_data - T.max(predict_data, axis=-1, keepdims=True)
    predict_data = predict_data - T.log(T.sum(T.exp(predict_data), axis=-1, keepdims=True))
    train_predict_cost = -T.sum(T.mul(one_hot_target_data, predict_data), axis=-1)
    train_predict_cost = train_predict_cost*T.flatten(target_mask, 1)
    train_model_cost = train_predict_cost.sum()/num_seqs
    train_frame_cost = train_predict_cost.sum()/target_mask.sum()

    # get regularizer cost
    train_regularizer_cost = regularize_network_params(network, penalty=l2)

    # get network parameters
    network_params = get_all_params(network, trainable=True)

    # get network gradients
    network_grads = theano.grad(cost=train_model_cost + train_regularizer_cost*l2_lambda,
                                wrt=network_params)

    if grad_max_norm>0.:
        network_grads, network_grads_norm = total_norm_constraint(tensor_vars=network_grads,
                                                                  max_norm=grad_max_norm,
                                                                  return_norm=True)
    else:
        network_grads_norm = T.sqrt(sum(T.sum(grad**2) for grad in network_grads))

    # set updater
    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_updates, trainer_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    skip_comp_list = []
    for rand_layer in rand_layer_list:
        skip_comp_list.append(T.sum(rand_layer.skip_comp*input_mask)/T.sum(input_mask))

    # get training (update) function
    training_fn = theano.function(inputs=[input_data,
                                          input_mask,
                                          target_data,
                                          target_mask],
                                  outputs=[train_frame_cost,
                                           network_grads_norm] + skip_comp_list,
                                  updates=train_updates)
    return training_fn, trainer_params

def set_network_predictor(input_data,
                          input_mask,
                          target_data,
                          target_mask,
                          num_outputs,
                          network):
    # get one hot target
    one_hot_target_data = T.extra_ops.to_one_hot(y=T.flatten(target_data, 1),
                                                 nb_class= num_outputs,
                                                 dtype=floatX)

    # get network output data
    predict_data = get_output(network, deterministic=True)

    # get prediction index
    predict_idx = T.argmax(predict_data, axis=-1)

    # get prediction cost
    predict_data = T.reshape(x=predict_data,
                             newshape=(-1, predict_data.shape[-1]),
                             ndim=2)

    predict_data = predict_data - T.max(predict_data, axis=-1, keepdims=True)
    predict_data = predict_data - T.log(T.sum(T.exp(predict_data), axis=-1, keepdims=True))
    predict_cost = -T.sum(T.mul(one_hot_target_data, predict_data), axis=-1)
    predict_cost = predict_cost*T.flatten(target_mask, 1)
    predict_cost = predict_cost.sum()/target_mask.sum()

    # get prediction function
    predict_fn = theano.function(inputs=[input_data,
                                         input_mask,
                                         target_data,
                                         target_mask],
                                 outputs=[predict_idx,
                                          predict_cost])

    return predict_fn

def network_evaluation(predict_fn,
                       data_stream):

    data_iterator = data_stream.get_epoch_iterator()

    # evaluation results
    total_nll = 0.
    total_fer = 0.
    total_cnt = 0.

    # for each batch
    for i, data in enumerate(data_iterator):
        # get input data
        input_data = data[0].astype(floatX)
        input_mask = data[1].astype(floatX)

        # get target data
        target_data = data[2]
        target_mask = data[3].astype(floatX)

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
        match_avg = numpy.sum(match_data)/numpy.sum(target_mask)

        # add up cost
        total_nll += predict_cost
        total_fer += (1.0 - match_avg)
        total_cnt += 1.

    total_nll /= total_cnt
    total_bpc = total_nll/numpy.log(2.0)
    total_fer /= total_cnt

    return total_nll, total_bpc, total_fer


def main(options):
    print 'Build and compile network'
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')

    skip_scale = theano.shared(convert_to_floatX(options['skip_scale']))

    network, rand_layer_list = build_network(input_data=input_data,
                                             input_mask=input_mask,
                                             num_inputs=options['num_inputs'],
                                             num_units_list=options['num_units_list'],
                                             num_outputs=options['num_outputs'],
                                             stochastic=options['stochastic'],
                                             skip_scale=skip_scale,
                                             dropout_ratio=options['dropout_ratio'],
                                             weight_noise=options['weight_noise'],
                                             use_layer_norm=options['use_layer_norm'],
                                             peepholes=options['peepholes'],
                                             learn_init=options['learn_init'],
                                             grad_clipping=options['grad_clipping'],
                                             gradient_steps=options['gradient_steps'],
                                             use_projection=options['use_projection'])

    network_params = get_all_params(network, trainable=True)

    print("number of parameters in model: %d" % count_params(network, trainable=True))

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
                                                      rand_layer_list=rand_layer_list,
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
                                      norm_path=options['norm_data_path'],
                                      which_set='train_si84',
                                      batch_size=options['batch_size'])

    print 'Start training'
    if os.path.exists(options['save_path'] + '_eval_history.npz'):
        evaluation_history = numpy.load(options['save_path'] + '_eval_history.npz')['eval_history'].tolist()
    else:
        evaluation_history = [[[10.0, 10.0, 1.0], [10.0, 10.0, 1.0]]]
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
                skip_means = train_output[2:]

                # show intermediate result
                if total_batch_cnt%options['train_disp_freq'] == 0 and total_batch_cnt!=0:
                    # pdb.set_trace()
                    best_idx = numpy.asarray(evaluation_history)[:, 1, 2].argmin()
                    print '============================================================================================'
                    print 'Model Name: ', options['save_path'].split('/')[-1]
                    print '============================================================================================'
                    print 'Epoch: ', str(e_idx), ', Update: ', str(total_batch_cnt)
                    print '--------------------------------------------------------------------------------------------'
                    print 'Prediction Cost: ', str(train_predict_cost)
                    print 'Gradient Norm: ', str(network_grads_norm)
                    print '--------------------------------------------------------------------------------------------'
                    print 'Skip Ratio: ', skip_means
                    print 'Skip Scale: ', str(skip_scale.get_value())
                    print '--------------------------------------------------------------------------------------------'
                    print 'Train NLL: ', str(evaluation_history[-1][0][0]), ', BPC: ', str(evaluation_history[-1][0][1]), ', FER: ', str(evaluation_history[-1][0][2])
                    print 'Valid NLL: ', str(evaluation_history[-1][1][0]), ', BPC: ', str(evaluation_history[-1][1][1]), ', FER: ', str(evaluation_history[-1][1][2])
                    print '--------------------------------------------------------------------------------------------'
                    print 'Best NLL: ', str(evaluation_history[best_idx][1][0]), ', BPC: ', str(evaluation_history[best_idx][1][1]), ', FER: ', str(evaluation_history[best_idx][1][2])

                # evaluation
                if total_batch_cnt%options['train_eval_freq'] == 0 and total_batch_cnt!=0:
                    train_eval_datastream = get_datastream(path=options['data_path'],
                                                           norm_path=options['norm_data_path'],
                                                           which_set='train_si84',
                                                           batch_size=options['eval_batch_size'])
                    valid_eval_datastream = get_datastream(path=options['data_path'],
                                                           norm_path=options['norm_data_path'],
                                                           which_set='test_dev93',
                                                           batch_size=options['eval_batch_size'])
                    train_nll, train_bpc, train_fer = network_evaluation(predict_fn,
                                                                         train_eval_datastream)
                    valid_nll, valid_bpc, valid_fer = network_evaluation(predict_fn,
                                                                         valid_eval_datastream)

                    # check over-fitting
                    if valid_fer>numpy.asarray(evaluation_history)[:, 1, 2].min():
                        early_stop_cnt += 1.
                    else:
                        early_stop_cnt = 0.
                        best_network_params_vals = get_model_param_values(network_params)
                        pickle.dump(best_network_params_vals,
                                    open(options['save_path'] + '_best_model.pkl', 'wb'))

                    if early_stop_cnt>10:
                        early_stop_flag = True
                        break

                    # save results
                    evaluation_history.append([[train_nll, train_bpc, train_fer],
                                               [valid_nll, valid_bpc, valid_fer]])
                    numpy.savez(options['save_path'] + '_eval_history',
                                eval_history=evaluation_history)

                # save network
                if total_batch_cnt%options['train_save_freq'] == 0 and total_batch_cnt!=0:
                    cur_network_params_val = get_model_param_values(network_params)
                    cur_trainer_params_val = get_update_params_values(trainer_params)
                    cur_total_batch_cnt = total_batch_cnt
                    pickle.dump([cur_network_params_val, cur_trainer_params_val, cur_total_batch_cnt],
                                open(options['save_path'] + '_last_model.pkl', 'wb'))

                if total_batch_cnt % 1000 == 0 and total_batch_cnt != 0:
                    skip_scale.set_value(convert_to_floatX(skip_scale.get_value() * 1.01))

            if early_stop_flag:
                break

    except KeyboardInterrupt:
        print 'Training Interrupted'
        cur_network_params_val = get_model_param_values(network_params)
        cur_trainer_params_val = get_update_params_values(trainer_params)
        cur_total_batch_cnt = total_batch_cnt
        pickle.dump([cur_network_params_val, cur_trainer_params_val, cur_total_batch_cnt],
                    open(options['save_path'] + '_last_model.pkl', 'wb'))

if __name__ == '__main__':
    from libs.lasagne_libs.updates import momentum
    parser = ArgumentParser()

    parser.add_argument('-b', '--batch_size', action='store',help='batch size', default=1)
    parser.add_argument('-n', '--num_layers', action='store',help='num of layers', default=5)
    parser.add_argument('-l', '--learn_rate', action='store', help='learning rate', default=4)
    parser.add_argument('-g', '--grad_norm', action='store', help='gradient norm', default=0.0)
    parser.add_argument('-c', '--grad_clipping', action='store', help='gradient clipping', default=1.0)
    parser.add_argument('-s', '--grad_steps', action='store', help='gradient steps', default=-1)
    parser.add_argument('-w', '--weight_noise', action='store', help='weight noise', default=0.0)
    parser.add_argument('-p', '--peepholes', action='store', help='peepholes', default=1)
    parser.add_argument('-r', '--reg_l2', action='store', help='l2 regularizer', default=0)
    parser.add_argument('-j', '--projection', action='store', help='projection layer', default=0)

    args = parser.parse_args()
    batch_size = int(args.batch_size)
    num_layers = int(args.num_layers)
    learn_rate= int(args.learn_rate)
    grad_norm = float(args.grad_norm)
    grad_clipping = float(args.grad_clipping)
    gradient_steps = int(args.grad_steps)
    weight_noise = float(args.weight_noise)
    peepholes = int(args.peepholes)
    l2_lambda = int(args.reg_l2)
    use_projection = int(args.projection)

    options = OrderedDict()

    options['skip_scale'] = 1.0
    options['stochastic'] = False

    options['num_inputs'] = 123
    options['num_units_list'] = [500]*num_layers
    options['num_outputs'] = 3436
    options['use_projection'] = True if use_projection==1 else False

    options['dropout_ratio'] = 0.0
    options['weight_noise'] = weight_noise
    options['use_layer_norm'] = False

    options['peepholes'] = True if peepholes==1 else False
    options['learn_init'] = False

    options['updater'] = momentum
    options['lr'] = 10**(-learn_rate)
    options['grad_norm'] = grad_norm
    options['grad_clipping'] = grad_clipping
    options['gradient_steps'] = gradient_steps
    options['l2_lambda'] = 0.0 if l2_lambda==0 else 10**(-l2_lambda)

    options['batch_size'] = batch_size
    options['eval_batch_size'] = 64
    options['num_epochs'] = 200

    options['train_disp_freq'] = 50
    options['train_eval_freq'] = 500
    options['train_save_freq'] = 100

    options['data_path'] = '/home/kimts/data/speech/wsj_fbank123.h5'
    options['norm_data_path'] = '/home/kimts/data/speech/wsj_fbank123_norm_data.npz'

    options['save_path'] = './wsj_deep_diff_skip_lstm' + \
                           '_lr' + str(int(learn_rate)) + \
                           '_gn' + str(int(grad_norm)) + \
                           '_gc' + str(int(grad_clipping)) + \
                           '_gs' + str(int(gradient_steps)) + \
                           '_nl' + str(int(num_layers)) + \
                           '_rg' + str(int(l2_lambda)) + \
                           '_p' + str(int(peepholes)) + \
                           '_j' + str(int(use_projection)) + \
                           '_b' + str(int(batch_size))


    reload_path = options['save_path'] + '_last_model.pkl'

    if os.path.exists(reload_path):
        options['reload_model'] = reload_path
    else:
        options['reload_model'] = None

    for key, val in options.iteritems():
        print str(key), ': ', str(val)

    main(options)