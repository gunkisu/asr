from lasagne import nonlinearities
from lasagne.layers import (InputLayer,
                            DropoutLayer,
                            ConcatLayer)
from libs.lasagne_libs.layers import SequenceDenseLayer
from libs.lasagne_libs.gating_layers import SkipLSTMLayer

def deep_bidir_skip_lstm_model(input_var,
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

    rand_layer_list = []

    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #######################################
    # deep bi-directional skip lstm layer #
    #######################################
    prev_input_layer = input_layer
    for num_units in num_units_list:
        # drop out
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout_ratio)

        # forward lstm
        lstm_fwd_layer = SkipLSTMLayer(input_data_layer=prev_input_layer,
                                       input_mask_layer=mask_layer,
                                       num_units=num_units,
                                       gradient_steps=gradient_steps,
                                       grad_clipping=grad_clipping,
                                       backwards=False)
        rand_layer_list.append(lstm_fwd_layer)

        # backward lstm
        lstm_bwd_layer = SkipLSTMLayer(input_data_layer=prev_input_layer,
                                       input_mask_layer=mask_layer,
                                       num_units=num_units,
                                       gradient_steps=gradient_steps,
                                       grad_clipping=grad_clipping,
                                       backwards=True)
        rand_layer_list.append(lstm_bwd_layer)

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

    return output_layer, rand_layer_list