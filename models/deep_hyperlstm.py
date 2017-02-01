from lasagne import nonlinearities
from lasagne.layers import InputLayer, ConcatLayer
from libs.lasagne_libs.hyper_layers import HyperLSTMLayer
from libs.lasagne_libs.layers import build_sequence_dense_layer

def build_deep_hyperlstm(input_var, mask_var, input_dim,
        num_layers, num_units, num_hyper_units, num_proj_units, 
        output_dim, grad_clipping, bidir=True):

    input_layer = InputLayer(shape=(None, None, input_dim),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
        prev_fwd_layer = HyperLSTMLayer(prev_input_layer,
                            num_units, num_hyper_units, num_proj_units,
                            mask_input=mask_layer, backwards=False, grad_clipping=grad_clipping)
        if bidir:
            prev_bwd_layer = HyperLSTMLayer(prev_input_layer,
                                num_units, num_hyper_units, num_proj_units,
                                mask_input=mask_layer, backwards=True, grad_clipping=grad_clipping)
     
            prev_input_layer = ConcatLayer(incomings=[prev_fwd_layer, prev_bwd_layer],
                                           axis=-1)
        else:
            prev_input_layer = prev_fwd_layer

    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim)
