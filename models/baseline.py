from lasagne.layers import InputLayer, ConcatLayer
from lasagne import nonlinearities
from libs.lasagne_libs.layers import SequenceDenseLayer
from libs.lasagne_libs.recurrent_layers import *

def deep_bidir_model(input_var,
                     mask_var,
                     num_inputs,
                     num_outputs,
                     num_units_list,
                     num_factors_list=None,
                     rnn_layer=LSTMLayer,
                     peepholes=False,
                     learn_init=False,
                     grad_clipping=0.0,
                     gradient_steps=-1,
                     use_softmax=True):

    if num_factors_list is None:
        num_factors_list = [num_units/2 for num_units in num_units_list]

    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ##############################
    # deep bi-directional layers #
    ##############################
    prev_input_layer = input_layer
    for num_factors, num_units in zip(num_factors_list, num_units_list):
        # forward rnn
        fwd_layer = rnn_layer(incoming=prev_input_layer,
                              mask_input=mask_layer,
                              num_units=num_units,
                              num_factors=num_factors,
                              peepholes=peepholes,
                              grad_clipping=grad_clipping,
                              gradient_steps=gradient_steps,
                              backwards=False)

        # backward rnn
        bwd_layer = rnn_layer(incoming=prev_input_layer,
                              mask_input=mask_layer,
                              num_units=num_units,
                              num_factors=num_factors,
                              peepholes=peepholes,
                              grad_clipping=grad_clipping,
                              gradient_steps=gradient_steps,
                              backwards=False)
        # concatenate forward/backward
        prev_input_layer = ConcatLayer(incomings=[fwd_layer, bwd_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      mask_input=mask_layer,
                                      num_outputs=num_outputs,
                                      W=init.Orthogonal(),
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    return output_layer