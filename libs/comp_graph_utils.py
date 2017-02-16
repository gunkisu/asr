import numpy
import theano
import lasagne

from theano import tensor as T
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import categorical_crossentropy

import itertools
import data.wsj.fuel_utils as fuel_utils

floatX = theano.config.floatX


def compute_loss(network, target_data, target_mask):
    o = get_output(network, deterministic=False)
    num_seqs = o.shape[0]
    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))

    ce = ce * T.flatten(target_mask, 1)

    ce_cost = ce.sum()/num_seqs
    ce_frame = ce.sum()/target_mask.sum()

    return ce_cost, ce_frame

def trainer(input_data, input_mask, target_data, target_mask, 
        network, updater, learning_rate, load_updater_params=None,
        ivector_data=None):

    ce_cost, ce_frame = compute_loss(network, target_data, target_mask)

    network_params = get_all_params(network, trainable=True)
    network_grads = theano.grad(cost=ce_cost,
                                wrt=network_params)

    network_grads_norm = T.sqrt(sum(T.sum(grad**2) for grad in network_grads))

    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_updates, trainer_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask]
    else:
        inputs = [input_data, input_mask, target_data, target_mask]

    outputs = [ce_frame, network_grads_norm]

    training_fn = theano.function(
            inputs=inputs, outputs=outputs, updates=train_updates)
     
    return training_fn, trainer_params

def trainer_lhuc(input_data,
                        input_mask,
                        target_data,
                        target_mask,
                        speaker_data,
                        network,
                        updater,
                        learning_rate,
                        load_updater_params=None):
    
    ce_cost, ce_frame = compute_loss(network, target_data, target_mask)

    network_params = get_all_params(network, trainable=True, speaker_independent=False)
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
                                          target_mask, speaker_data],
                                  outputs=[ce_frame,
                                           network_grads_norm],
                                  updates=train_updates)
    return training_fn, trainer_params

def predictor(input_data, input_mask, target_data, target_mask, network, ivector_data=None):
    o = get_output(network, deterministic=False)
    num_seqs = o.shape[0]
    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))

    pred_idx = T.argmax(o, axis=-1)
    ce = ce * T.flatten(target_mask, 1)

    ce_frame = ce.sum()/target_mask.sum()

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask]
    else:
        inputs = [input_data, input_mask, target_data, target_mask]
    outputs = [pred_idx, ce_frame]

    fn = theano.function(inputs=inputs, outputs=outputs)

    return fn

def predictor_lhuc(input_data,
                          input_mask,
                          target_data,
                          target_mask,
                          speaker_data,
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
                                         target_mask, speaker_data],
                                 outputs=[pred_idx,
                                          ce_frame])

def eval_net(predict_fn, data_stream, use_ivectors=False):

    data_iterator = data_stream.get_epoch_iterator()

    total_nll = 0.
    total_fer = 0.

    for batch_cnt, data in enumerate(data_iterator, start=1):
        input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = data
     
        if use_ivectors:
            predict_output = predict_fn(input_data,
                                    input_mask,
                                    ivector_data,
                                    target_data,
                                    target_mask)
        else:
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

        break

    total_nll /= batch_cnt 
    total_fer /= batch_cnt

    return total_nll, total_fer

def eval_net_lhuc(predict_fn,
                       data_stream, spk_stream, spk_list):

    total_nll = 0.
    total_fer = 0.

    for batch_cnt, lhuc_data in enumerate(itertools.izip(data_stream.get_epoch_iterator(),
            spk_stream.get_epoch_iterator()), start=1):
        data, spk_data = lhuc_data
        input_data, input_mask, target_data, target_mask = data

        predict_output = predict_fn(input_data,
                                    input_mask,
                                    target_data,
                                    target_mask, fuel_utils.spk_to_ids(spk_list, spk_data[0]))
        predict_idx = predict_output[0]
        predict_cost = predict_output[1]

        match_data = (target_data == predict_idx)*target_mask
        match_avg = numpy.sum(match_data)/numpy.sum(target_mask)

        total_nll += predict_cost
        total_fer += (1.0 - match_avg)

    total_nll /= batch_cnt 
    total_fer /= batch_cnt

    return total_nll, total_fer


