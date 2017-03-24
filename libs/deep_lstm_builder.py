from lasagne import nonlinearities
from lasagne.layers import (InputLayer,
                            DropoutLayer,
                            ConcatLayer)
from libs.lhuclstm_layers import LSTMPLayer, LSTMLayer

from lasagne.layers import DenseLayer, ReshapeLayer, reshape, Gate
from lasagne.layers import get_output_shape

from models.utils import build_sequence_dense_layer, build_input_layer, build_ivector_layer, concatenate_layers

def build_deep_lstm(input_var, mask_var, input_dim, num_layers, num_units, output_dim,
                          grad_clipping, is_bidir, ivector_dim, ivector_var=None):
    
    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)
    
    if ivector_var:
        ivector_layer = build_ivector_layer(ivector_dim, ivector_var)
        input_layer = concatenate_layers(input_layer, ivector_layer)

    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
        fwd_layer = LSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          grad_clipping=grad_clipping,
                                          backwards=False)
        if is_bidir:
            bwd_layer = LSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          grad_clipping=grad_clipping,
                                          backwards=True)

            prev_input_layer = ConcatLayer(incomings=[fwd_layer, bwd_layer],
                                       axis=-1)
        else:
            prev_input_layer = fwd_layer


    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim)


def build_deep_lstmp(input_var, mask_var, input_dim, num_layers, num_units, num_proj_units, output_dim,
                          grad_clipping, is_bidir, ivector_dim,  
                          ivector_var=None):
    
    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)
    
    if ivector_var:
        ivector_layer = build_ivector_layer(ivector_dim, ivector_var)
        input_layer = concatenate_layers(input_layer, ivector_layer)

    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
        fwd_layer = LSTMPLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          num_proj_units=num_proj_units,
                                          grad_clipping=grad_clipping,
                                          backwards=False)
        if is_bidir:
            bwd_layer = LSTMPLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          num_proj_units=num_proj_units,
                                          grad_clipping=grad_clipping,
                                          backwards=True)

            prev_input_layer = ConcatLayer(incomings=[fwd_layer, bwd_layer],
                                       axis=-1)
        else:
            prev_input_layer = fwd_layer


    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim)

