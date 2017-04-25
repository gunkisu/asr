from collections import OrderedDict
import numpy
import theano
import lasagne

from theano import tensor as T
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import categorical_crossentropy

from libs.utils import gen_win, skip_frames

import itertools

from theano.ifelse import ifelse

floatX = theano.config.floatX

def delayed(o, target_data, target_mask, delay):
    return o[:,delay:,:], target_data[:,delay:], target_mask[:,delay:]

def delayed_tbptt(o, target_data, target_mask, is_first_win, delay):
    return ifelse(is_first_win, o[:,delay:,:], o), \
        ifelse(is_first_win, target_data[:,delay:], target_data), \
        ifelse(is_first_win, target_mask[:,delay:], target_mask)

def context_tbptt(o, target_data, target_mask, context):
    return o[:,:context,:], target_data[:,:context], target_mask[:,:context]

def compute_loss(network, target_data, target_mask, delay):
    o = get_output(network, deterministic=False)
        
    n_batch, n_seq, n_feat  = o.shape
   
    o, target_data, target_mask = delayed(o, target_data, target_mask, delay)
    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))

    
    ce = ce * T.flatten(target_mask, 1)
    ce_cost = ce.sum()/n_batch
    ce_frame_sum = ce.sum()

    pred_idx = T.argmax(o, axis=-1)

    return ce_cost, ce_frame_sum, pred_idx

def compute_loss_skip(network, target_data, target_mask, skip):
    o = get_output(network, deterministic=False)
        
    n_batch, n_seq, n_feat  = o.shape
    _, n_target_seq = target_data.shape
        
    o = o.repeat(skip, axis=1)
    o = o[:,:n_target_seq,:]
   
    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))

    ce = ce * T.flatten(target_mask, 1)
    ce_cost = ce.sum()/n_batch
    ce_frame_sum = ce.sum()

    pred_idx = T.argmax(o, axis=-1)

    return ce_cost, ce_frame_sum, pred_idx


def compute_loss_tbptt(network, target_data, target_mask, is_first_win, delay, context):
    o = get_output(network, deterministic=False)
        
    n_batch, n_seq, n_feat  = o.shape
  
    if delay:
        o, target_data, target_mask = delayed_tbptt(o, target_data, target_mask, is_first_win, delay)
    elif context:
        o, target_data, target_mask = context_tbptt(o, target_data, target_mask, context)

    ce = categorical_crossentropy(predictions=T.reshape(o, (-1, o.shape[-1]), ndim=2), 
            targets=T.flatten(target_data, 1))
    
    ce = ce * T.flatten(target_mask, 1)
    ce_cost = ce.sum()/n_batch
    ce_frame_sum = ce.sum()

    pred_idx = T.argmax(o, axis=-1)

    return ce_cost, ce_frame_sum, pred_idx

def trainer(input_data, input_mask, target_data, target_mask, network, updater, 
        learning_rate, delay, load_updater_params=None, ivector_data=None):

    ce_cost, ce_frame_sum, _ = compute_loss(network, target_data, target_mask, delay)
  
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

    outputs = [ce_frame_sum, network_grads_norm]

    training_fn = theano.function(
            inputs=inputs, outputs=outputs, updates=train_total_updates)
     
    return training_fn, trainer_params

def trainer_skip(input_data, input_mask, target_data, target_mask, network, updater, 
        learning_rate, skip, load_updater_params=None, ivector_data=None):

    ce_cost, ce_frame_sum, _ = compute_loss_skip(network, target_data, target_mask, skip)
  
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

    outputs = [ce_frame_sum, network_grads_norm]

    training_fn = theano.function(
            inputs=inputs, outputs=outputs, updates=train_total_updates)
     
    return training_fn, trainer_params

def trainer_tbptt(input_data, input_mask, target_data, target_mask, network, updater, 
        learning_rate, tbptt_layers, is_first_win, delay, context, load_updater_params=None, ivector_data=None):

    ce_cost, ce_frame_sum, _ = compute_loss_tbptt(network, target_data, target_mask, is_first_win, delay, context)
  
    network_params = get_all_params(network, trainable=True)
    network_grads = theano.grad(cost=ce_cost,
                                wrt=network_params)

    network_grads_norm_sum = sum(T.sum(grad**2) for grad in network_grads)

    train_total_updates = OrderedDict()

    train_lr = theano.shared(lasagne.utils.floatX(learning_rate))
    train_updates, trainer_params = updater(loss_or_grads=network_grads,
                                            params=network_params,
                                            learning_rate=train_lr,
                                            load_params_dict=load_updater_params)

    train_total_updates.update(train_updates)

    for l in tbptt_layers:
        train_total_updates.update(l.get_updates())

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask, is_first_win]
    else:
        inputs = [input_data, input_mask, target_data, target_mask, is_first_win]

    outputs = [ce_frame_sum, network_grads_norm_sum]

    training_fn = theano.function(
            inputs=inputs, outputs=outputs, updates=train_total_updates, on_unused_input='warn')
     
    return training_fn, trainer_params

def predictor_tbptt(input_data, input_mask, target_data, target_mask, network, 
        tbptt_layers, is_first_win, delay, context, ivector_data=None):
    _, ce_frame_sum, pred_idx = compute_loss_tbptt(network, target_data, target_mask, is_first_win, delay, context)

    predict_updates = OrderedDict()

    for l in tbptt_layers:
        predict_updates.update(l.get_updates())

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask, is_first_win]
    else:
        inputs = [input_data, input_mask, target_data, target_mask, is_first_win]
    outputs = [ce_frame_sum, pred_idx]

    fn = theano.function(inputs=inputs, outputs=outputs, updates=predict_updates, on_unused_input='warn')
    return fn

def predictor(input_data, input_mask, target_data, target_mask, network, delay, ivector_data=None):
    _, ce_frame_sum, pred_idx = compute_loss(network, target_data, target_mask, delay)

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask]
    else:
        inputs = [input_data, input_mask, target_data, target_mask]
    outputs = [ce_frame_sum, pred_idx]

    fn = theano.function(inputs=inputs, outputs=outputs)
    return fn

def predictor_skip(input_data, input_mask, target_data, target_mask, network, skip, ivector_data=None):
    _, ce_frame_sum, pred_idx = compute_loss_skip(network, target_data, target_mask, skip)

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, target_data, target_mask]
    else:
        inputs = [input_data, input_mask, target_data, target_mask]
    outputs = [ce_frame_sum, pred_idx]

    fn = theano.function(inputs=inputs, outputs=outputs)
    return fn

def eval_net(predict_fn, data_stream, use_ivectors=False, delay=0):
    data_iterator = data_stream.get_epoch_iterator()

    total_ce_sum = 0.
    total_accuracy_sum = 0
    total_frame_count = 0
        
    for batch_cnt, data in enumerate(data_iterator, start=1):
        input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = data

        if use_ivectors:
            predict_output = predict_fn(input_data, input_mask, ivector_data, target_data, target_mask)
        else:
            predict_output = predict_fn(input_data, input_mask, target_data, target_mask)

        ce_frame_sum, pred_idx = predict_output
        
        target_data, target_mask = target_data[:,delay:], target_mask[:,delay:] 
        match_data = (target_data == pred_idx)*target_mask

        total_accuracy_sum += match_data.sum()
        total_ce_sum += ce_frame_sum
        total_frame_count += target_mask.sum()

    avg_ce = total_ce_sum / total_frame_count
    avg_fer = 1.0 - (total_accuracy_sum / total_frame_count)

    return avg_ce, avg_fer

def eval_net_skip(predict_fn, data_stream, skip, skip_random, use_ivectors=False):
    data_iterator = data_stream.get_epoch_iterator()

    total_ce_sum = 0.
    total_accuracy_sum = 0
    total_frame_count = 0
        
    for batch_cnt, data in enumerate(data_iterator, start=1):
        input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = data
        s_input_data, s_input_mask, s_ivector_data, s_ivector_mask, _, _ = \
            skip_frames(data, skip, skip_random)        

        if use_ivectors:
            predict_output = predict_fn(s_input_data, s_input_mask, s_ivector_data, target_data, target_mask)
        else:
            predict_output = predict_fn(s_input_data, s_input_mask, target_data, target_mask)

        ce_frame_sum, pred_idx = predict_output
        
        match_data = (target_data == pred_idx)*target_mask

        total_accuracy_sum += match_data.sum()
        total_ce_sum += ce_frame_sum
        total_frame_count += target_mask.sum()

    avg_ce = total_ce_sum / total_frame_count
    avg_fer = 1.0 - (total_accuracy_sum / total_frame_count)

    return avg_ce, avg_fer


def eval_net_tbptt(predict_fn, data_stream, tbptt_layers, num_tbptt_steps, batch_size, right_context, use_ivectors=False, delay=0):
    data_iterator = data_stream.get_epoch_iterator()

    total_accuracy_sum = 0
    total_ce_sum = 0.
    total_frame_count = 0
    for b_idx, batch in enumerate(data_iterator, start=1):

        i_data, _, _, _, t_data, t_mask = batch
        n_batch, _, _ = i_data.shape

        for l in tbptt_layers:
            l.reset(n_batch)

        ce_frame = 0.0
        pred_idx_list = []
        for win_idx, win in enumerate(gen_win(batch, num_tbptt_steps, right_context), start=1):
            input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = win
            is_first_win = 1 if win_idx == 1 else 0

            if use_ivectors:
                predict_output = predict_fn(input_data, input_mask, ivector_data, target_data, target_mask, is_first_win)
            else:
                predict_output = predict_fn(input_data, input_mask, target_data, target_mask, is_first_win)

            ce_frame_sum, pred_idx = predict_output

            total_ce_sum += ce_frame_sum
            pred_idx_list.append(pred_idx)

        target_data, target_mask = t_data[:,delay:], t_mask[:,delay:] 

        pred_idx = numpy.concatenate(pred_idx_list, axis=1)
        match_data = (target_data == pred_idx)*target_mask

        total_accuracy_sum += match_data.sum()
        total_frame_count += target_mask.sum() 

    avg_ce = total_ce_sum / total_frame_count
    avg_fer = 1.0 - (total_accuracy_sum / total_frame_count)

    return avg_ce, avg_fer

def ff(network, input_data, input_mask, ivector_data=None):
    predict_data = get_output(network, deterministic=True)

    
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data]
    else:
        inputs = [input_data, input_mask]

    predict_fn = theano.function(inputs=inputs,
                                 outputs=[predict_data])

    return predict_fn

def ff_tbptt(input_data, input_mask, network, is_first_win, delay, num_tbptt_steps, ivector_data=None):
    o = get_output(network, deterministic=False)
        
    if delay:
        o = ifelse(is_first_win, o[:,delay:,:], o)
    elif num_tbptt_steps:
        o = o[:,:num_tbptt_steps,:]

    inputs = None
    if ivector_data:
        inputs = [input_data, input_mask, ivector_data, is_first_win]
    else:
        inputs = [input_data, input_mask, is_first_win]
    outputs = [o]

    fn = theano.function(inputs=inputs, outputs=outputs, on_unused_input='warn')
    return fn
