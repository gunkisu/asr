from lasagne import nonlinearities
from lasagne.layers import (InputLayer,
                            DropoutLayer,
                            ConcatLayer)
from libs.lasagne.layers import SequenceDenseLayer
from libs.lasagne.layers import LSTMLayer

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
                                      mask_input=mask_layer,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    return output_layer




