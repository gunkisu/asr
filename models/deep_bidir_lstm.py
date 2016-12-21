from lasagne import nonlinearities
from lasagne.layers import (InputLayer,
                            DropoutLayer,
                            ConcatLayer)
from libs.lasagne_libs.layers import SequenceDenseLayer
from libs.lasagne_libs.layers import LSTMLayer, LSTMPLayer
from lasagne.layers import LSTMLayer as LasagneLSTMLayer
from lasagne.layers import DenseLayer, ReshapeLayer
from lasagne.layers import get_output_shape

def deep_bidir_lstm_model(input_var,
                          mask_var,
                          num_inputs,
                          num_units_list,
                          num_outputs,
                          dropout_ratio=0.2,
                          weight_noise=0.0,
                          use_layer_norm=True,
                          peepholes=False,
                          learn_init=False,
                          grad_clipping=0.0,
                          gradient_steps=-1,
                          use_softmax=True,
                          use_projection=False):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ##################################
    # deep bi-directional lstm layer #
    ##################################
    prev_input_layer = input_layer
    for num_units in num_units_list:
        # drop out
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout_ratio)

        # forward lstm
        lstm_fwd_layer = LSTMLayer(incoming=prev_input_layer,
                                   mask_input=mask_layer,
                                   num_units=num_units,
                                   dropout_ratio=dropout_ratio,
                                   use_layer_norm=use_layer_norm,
                                   weight_noise=weight_noise,
                                   peepholes=peepholes,
                                   grad_clipping=grad_clipping,
                                   gradient_steps=gradient_steps,
                                   backwards=False)

        # backward lstm
        lstm_bwd_layer = LSTMLayer(incoming=prev_input_layer,
                                   mask_input=mask_layer,
                                   num_units=num_units,
                                   dropout_ratio=dropout_ratio,
                                   use_layer_norm=use_layer_norm,
                                   weight_noise=weight_noise,
                                   peepholes=peepholes,
                                   grad_clipping=grad_clipping,
                                   gradient_steps=gradient_steps,
                                   backwards=True)

        # concatenate forward/backward
        prev_input_layer = ConcatLayer(incomings=[lstm_fwd_layer, lstm_bwd_layer],
                                       axis=-1)

        if use_projection:
            prev_input_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                                  num_outputs=num_units)



    ################
    # output layer #
    ################
    output_layer = DropoutLayer(incoming=prev_input_layer,
                                p=dropout_ratio)
    output_layer = SequenceDenseLayer(incoming=output_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    return output_layer

def deep_bidir_lstm_prj_model(input_var,
                              mask_var,
                              num_inputs,
                              num_prj_list,
                              num_units_list,
                              num_outputs,
                              dropout_ratio=0.2,
                              weight_noise=0.0,
                              use_layer_norm=True,
                              peepholes=False,
                              learn_init=False,
                              grad_clipping=0.0,
                              gradient_steps=-1,
                              use_softmax=True):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ##################################
    # deep bi-directional lstm layer #
    ##################################
    prev_input_layer = input_layer
    for num_prj, num_units in zip(num_prj_list, num_units_list):
        # drop out
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout_ratio)

        # forward lstm
        lstm_fwd_layer = LSTMPLayer(incoming=prev_input_layer,
                                    mask_input=mask_layer,
                                    num_prj=num_prj,
                                    num_units=num_units,
                                    dropout_ratio=dropout_ratio,
                                    weight_noise=weight_noise,
                                    peepholes=peepholes,
                                    grad_clipping=grad_clipping,
                                    gradient_steps=gradient_steps,
                                    backwards=False)

        # backward lstm
        lstm_bwd_layer = LSTMPLayer(incoming=prev_input_layer,
                                    mask_input=mask_layer,
                                    num_prj=num_prj,
                                    num_units=num_units,
                                    dropout_ratio=dropout_ratio,
                                    weight_noise=weight_noise,
                                    peepholes=peepholes,
                                    grad_clipping=grad_clipping,
                                    gradient_steps=gradient_steps,
                                    backwards=True)

        # concatenate forward/backward
        prev_input_layer = ConcatLayer(incomings=[lstm_fwd_layer, lstm_bwd_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    output_layer = DropoutLayer(incoming=prev_input_layer,
                                p=dropout_ratio)
    output_layer = SequenceDenseLayer(incoming=output_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    return output_layer

def deep_bidir_lstm_alex(input_var,
                          mask_var,
                          input_dim,
                          num_units_list,
                          output_dim,
                          grad_clipping=1.0):
    
    input_layer = InputLayer(shape=(None, None, input_dim),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    prev_input_layer = input_layer
    for num_units in num_units_list:
        lstm_fwd_layer = LasagneLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          grad_clipping=grad_clipping,
                                          backwards=False)
        lstm_bwd_layer = LasagneLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          grad_clipping=grad_clipping,
                                          backwards=True)

        prev_input_layer = ConcatLayer(incomings=[lstm_fwd_layer, lstm_bwd_layer],
                                       axis=-1)



    n_batch, n_time_steps, _ = input_var.shape
    reshape_layer = ReshapeLayer(prev_input_layer, (-1, [2]))
    dense_layer = DenseLayer(reshape_layer, num_units=output_dim, nonlinearity=nonlinearities.softmax)
    output_layer = ReshapeLayer(dense_layer, (n_batch, n_time_steps, output_dim))

#    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
#                                      num_outputs=output_dim,
#                                      mask_input=mask_layer,
#                                      nonlinearity=nonlinearities.softmax)
    return output_layer

def deep_bidir_lstm_lhuc(input_var,
                          mask_var,
                          input_dim,
                          num_units_list,
                          output_dim,
                          grad_clipping=1.0):
    
    input_layer = InputLayer(shape=(None, None, input_dim),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    prev_input_layer = input_layer
    for num_units in num_units_list:
        lstm_fwd_layer = LSTMLayer(incoming=prev_input_layer,
                                   mask_input=mask_layer,
                                   num_units=num_units,
                                   grad_clipping=grad_clipping,
                                   backwards=False)
        lstm_bwd_layer = LSTMLayer(incoming=prev_input_layer,
                                   mask_input=mask_layer,
                                   num_units=num_units,
                                   grad_clipping=grad_clipping,
                                   backwards=True)

        prev_input_layer = ConcatLayer(incomings=[lstm_fwd_layer, lstm_bwd_layer],
                                       axis=-1)


    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=output_dim,
                                      mask_input=mask_layer,
                                      nonlinearity=nonlinearities.softmax)
    return output_layer


