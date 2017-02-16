from lasagne import nonlinearities
from lasagne.layers import (InputLayer,
                            DropoutLayer,
                            ConcatLayer)
from libs.lasagne_libs.layers import SequenceDenseLayer
from libs.lasagne_libs.layers import LSTMLayer, LSTMPLayer
from lasagne.layers import LSTMLayer as LasagneLSTMLayer
from lasagne.layers import DenseLayer, ReshapeLayer, reshape, Gate
from lasagne.layers import get_output_shape
from libs.lasagne_libs.lhuc_layers import LHUCLayer, exp, two_sigmoid
from libs.lasagne_libs.layers import build_sequence_dense_layer

from models.utils import build_input_layer, build_ivector_layer, concatenate_layers

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

def deep_bidir_lstm_share_model(input_var,
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
        lstm_fwd_layer = LasagneLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          peepholes=True,
                                          grad_clipping=grad_clipping,
                                          gradient_steps=gradient_steps,
                                          backwards=False)

        # backward lstm
        lstm_bwd_layer = LasagneLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          ingate=Gate(W_in=lstm_fwd_layer.W_in_to_ingate,
                                                      W_hid=lstm_fwd_layer.W_hid_to_ingate,
                                                      W_cell=lstm_fwd_layer.W_cell_to_ingate,
                                                      b=lstm_fwd_layer.b_ingate),
                                          forgetgate=Gate(W_in=lstm_fwd_layer.W_in_to_forgetgate,
                                                          W_hid=lstm_fwd_layer.W_hid_to_forgetgate,
                                                          W_cell=lstm_fwd_layer.W_cell_to_forgetgate,
                                                          b=lstm_fwd_layer.b_forgetgate),
                                          cell=Gate(W_in=lstm_fwd_layer.W_in_to_cell,
                                                    W_hid=lstm_fwd_layer.W_hid_to_cell,
                                                    W_cell=None,
                                                    b=lstm_fwd_layer.b_cell,
                                                    nonlinearity=nonlinearities.tanh),
                                          outgate=Gate(W_in=lstm_fwd_layer.W_in_to_outgate,
                                                       W_hid=lstm_fwd_layer.W_hid_to_outgate,
                                                       W_cell=lstm_fwd_layer.W_cell_to_outgate,
                                                       b=lstm_fwd_layer.b_outgate),
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

def build_deep_bidir_lstm_alex(input_var, mask_var, input_dim, num_units_list, output_dim,
                          grad_clipping=1.0, bidir=True, 
                          ivector_var=None, ivector_dim=100):
    
    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)
    
    if ivector_var:
        ivector_layer = build_ivector_layer(ivector_dim, ivector_var)
        input_layer = concatenate_layers(input_layer, ivector_layer)

    prev_input_layer = input_layer
    for num_units in num_units_list:
        lstm_fwd_layer = LasagneLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          grad_clipping=grad_clipping,
                                          backwards=False)
        if bidir:
            lstm_bwd_layer = LasagneLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          grad_clipping=grad_clipping,
                                          backwards=True)

            prev_input_layer = ConcatLayer(incomings=[lstm_fwd_layer, lstm_bwd_layer],
                                       axis=-1)
        else:
            prev_input_layer = lstm_fwd_layer


    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim)

def build_deep_bidir_lstm_lhuc(input_var,
                          mask_var,
                          input_dim,
                          num_units_list,
                          output_dim,
                          speaker_var,
                          num_speakers,
                          grad_clipping=1.0):
    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)

    speaker_input_layer = InputLayer(shape=(None,),
                            input_var=speaker_var)

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
        prev_input_layer = LHUCLayer(prev_input_layer,
                speaker_input_layer, num_speakers, two_sigmoid)

    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim)

