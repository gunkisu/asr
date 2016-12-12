import numpy, theano, lasagne, pickle, os
from theano import tensor as T
from collections import OrderedDict
from models.deep_bidir_lstm import deep_bidir_lstm_model
from libs.lasagne.utils import get_model_param_values, get_update_params_values
from libs.param_utils import set_model_param_value
from lasagne.layers import get_output, get_all_params
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import total_norm_constraint

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Padding, FilterSources, AgnosticTransformer

from six import iteritems

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

from fuel.transformers import AgnosticTransformer

class ConcatenateTransformer(AgnosticTransformer):
    def __init__(self, data_stream, concat_sources, new_source=None, **kwargs):
        if any(source not in data_stream.sources for source in concat_sources):
            raise ValueError("sources must all be contained in "
                             "data_stream.sources")

        self.new_source = new_source if not new_source else '_'.join(concat_sources)
        if data_stream.axis_labels:
            axis_labels = dict((source, labels) for (source, labels)
                    in iteritems(data_stream.axis_labels)
                        if source not in concat_sources) 
            axis_labels[self.new_source] = 'concatenated source'
            kwargs.setdefault('axis_labels', axis_labels)
        
        super(AgnosticTransformer, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples,
            **kwargs)

        new_sources = [s for s in data_stream.sources if s not in concat_sources]
        new_sources.append(self.new_source)
        self.sources = tuple(new_sources)
        self.concat_sources = concat_sources
       
    def transform_any(self, data):
        raise NotImplementedError()

def get_datastream(path, which_set='train_si84', batch_size=1, use_ivectors=True):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    print path, which_set
    iterator_scheme = ShuffledScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    if use_ivectors:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'ivectors', 'targets'])
        fs = ConcatenateTransformer(fs, ['features', 'ivectors'])
    else:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'targets'])
    padded_stream = Padding(data_stream=fs)
    return padded_stream

def build_network(input_data,
                  input_mask,
                  num_inputs,
                  num_units_list,
                  num_outputs,
                  dropout_ratio=0.2,
                  weight_noise=0.0,
                  use_layer_norm=True,
                  peepholes=False,
                  learn_init=True,
                  grad_clipping=0.0,
                  gradient_steps=-1):

    network = deep_bidir_lstm_model(input_var=input_data,
                                    mask_var=input_mask,
                                    num_inputs=num_inputs,
                                    num_units_list=num_units_list,
                                    num_outputs=num_outputs,
                                    dropout_ratio=dropout_ratio,
                                    weight_noise=weight_noise,
                                    use_layer_norm=use_layer_norm,
                                    peepholes=peepholes,
                                    learn_init=learn_init,
                                    grad_clipping=grad_clipping,
                                    gradient_steps=gradient_steps,
                                    use_softmax=False)
    return network

def set_network_trainer(input_data,
                        input_mask,
                        target_data,
                        target_mask,
                        num_outputs,
                        network,
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

    # get training (update) function
    training_fn = theano.function(inputs=[input_data,
                                          input_mask,
                                          target_data,
                                          target_mask],
                                  outputs=[train_frame_cost,
                                           network_grads_norm],
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
    predict_idx = T.argmax(T.exp(predict_data), axis=-1)

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

def eval_nete(predict_fn,
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

def save_network(network_params, trainer_params, total_batch_cnt, save_path):
    cur_network_params_val = get_model_param_values(network_params)
    cur_trainer_params_val = get_update_params_values(trainer_params)
    cur_total_batch_cnt = total_batch_cnt
    pickle.dump([cur_network_params_val, cur_trainer_params_val, cur_total_batch_cnt],
                open(save_path + '_last_model.pkl', 'wb'))


def show_status(save_path, e_idx, total_batch_cnt, train_predict_cost, network_grads_norm, evaluation_history):
    model = save_path.split('/')[-1]
#    print '============================================================================================'
    print '--'
    print 'Model Name: {}'.format(model)
#    print '============================================================================================'
    print 'Epoch: {}, Update: {}'.format(e_idx, total_batch_cnt)
#    print '--------------------------------------------------------------------------------------------'
    print 'Prediction Cost: {}'.format(train_predict_cost)
    print 'Gradient Norm: {}'.format(network_grads_norm)
#    print '--------------------------------------------------------------------------------------------'
#    print '--------------------------------------------------------------------------------------------'
#    print 'Train NLL: ', str(evaluation_history[-1][0][0]), ', BPC: ', str(evaluation_history[-1][0][1]), ', FER: ', str(evaluation_history[-1][0][2])
    print 'Valid NLL: ', str(evaluation_history[-1][1][0]), ', BPC: ', str(evaluation_history[-1][1][1]), ', FER: ', str(evaluation_history[-1][1][2])
    print '--'

