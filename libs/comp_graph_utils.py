from collections import OrderedDict
import numpy
import theano
import lasagne

from theano import tensor as T
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import categorical_crossentropy

import itertools

from theano.ifelse import ifelse

floatX = theano.config.floatX

def delayed(o, target_data, target_mask, delay):
    return o[:,delay:,:], target_data[:,delay:], target_mask[:,delay:]

def compute_loss(network, target_data, target_mask, delay=0):
    o = get_output(network, deterministic=False)
        
    n_batch, n_seq, n_feat  = o.shape
   
    o, target_data, target_mask = delayed(o, target_data, target_mask, delay)
    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))

    
    ce = ce * T.flatten(target_mask, 1)
    ce_cost = ce.sum()/n_batch
    ce_frame = ce.sum()/target_mask.sum()

    pred_idx = T.argmax(o, axis=-1)

    return ce_cost, ce_frame, pred_idx

def trainer(input_data, input_mask, target_data, target_mask, network, updater, 
        learning_rate, load_updater_params=None, ivector_data=None, delay=0):

    ce_cost, ce_frame, _ = compute_loss(network, target_data, target_mask, delay)
  
    network_params = get_all_params(network, trainable=True)
    network_grads = theano.grad(cost=ce_cost,
                                wrt=network_params)

    network_grads_norm = T.sqrt(sum(T.sum(grad**2) for grad in network_grads))

    train_total_updates = OrderedDict()

    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_updates, trainer_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    train_total_updates.update(train_updates)

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask]
    else:
        inputs = [input_data, input_mask, target_data, target_mask]

    outputs = [ce_frame, network_grads_norm]

    training_fn = theano.function(
            inputs=inputs, outputs=outputs, updates=train_total_updates)
     
    return training_fn, trainer_params

def predictor(input_data, input_mask, target_data, target_mask, network, ivector_data=None, delay=0):
    _, ce_frame, pred_idx = compute_loss(network, target_data, target_mask, delay)

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask]
    else:
        inputs = [input_data, input_mask, target_data, target_mask]
    outputs = [ce_frame, pred_idx]

    fn = theano.function(inputs=inputs, outputs=outputs)
    return fn

def eval_net(predict_fn, data_stream, batch_size, use_ivectors=False, delay=0):
    data_iterator = data_stream.get_epoch_iterator()

    total_nll = 0.
    total_fer = 0.
        
    for batch_cnt, data in enumerate(data_iterator, start=1):
        input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = data

        if use_ivectors:
            predict_output = predict_fn(input_data, input_mask, ivector_data, target_data, target_mask)
        else:
            predict_output = predict_fn(input_data, input_mask, target_data, target_mask)

        ce_frame, pred_idx = predict_output
        
        target_data, target_mask = target_data[:,delay:], target_mask[:,delay:] 
        match_data = (target_data == pred_idx)*target_mask
        fer = (1.0 - match_data.sum()/target_mask.sum())

        total_nll += ce_frame
        total_fer += fer

    total_nll /= batch_cnt 
    total_fer /= batch_cnt

    return total_nll, total_fer

def ff(network, input_data, input_mask, ivector_data=None):
    predict_data = get_output(network, deterministic=True)

    
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data]
    else:
        inputs = [input_data, input_mask]

    predict_fn = theano.function(inputs=inputs,
                                 outputs=[predict_data])

    return predict_fn