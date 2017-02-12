from lasagne import nonlinearities
from lasagne.layers import InputLayer, ConcatLayer, LSTMLayer
from libs.lasagne_libs.hyper_layers import HyperLSTMLayer, \
        HyperLHUCLSTMLayer, HyperTiedLHUCLSTMLayer

from libs.lasagne_libs.layers import build_sequence_dense_layer

def build_input_layer(input_dim, input_var, mask_var):
    input_layer = InputLayer(shape=(None, None, input_dim),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    
    return (input_layer, mask_layer)

def get_layer(layer_name, hyper_layer, prev_input_layer, 
        num_units, num_hyper_units, num_proj_units, 
        mask_layer, backwards, grad_clipping):

    if not hyper_layer:
        return LSTMLayer(prev_input_layer, num_units,
                                   mask_input=mask_layer,
                                   grad_clipping=grad_clipping,
                                   backwards=backwards)
                              

    if layer_name == 'HyperLSTMLayer':
        return HyperLSTMLayer(prev_input_layer,
                                num_units, num_hyper_units, num_proj_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping)
    elif layer_name == 'HyperLHUCLSTMLayer':
        return HyperLHUCLSTMLayer(prev_input_layer,
                                num_units, num_hyper_units, num_proj_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping)
    elif layer_name == 'HyperTiedLHUCLSTMLayer': 
        return HyperTiedLHUCLSTMLayer(prev_input_layer,
                                num_units, num_hyper_units, num_proj_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping)
    elif layer_name == 'HyperTiedLHUCOutLSTMLayer':
        return HyperTiedLHUCOutLSTMLayer(prev_input_layer,
                                num_units, num_hyper_units, num_proj_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping)
        

def build_deep_hyper_lstm(layer_name, input_var, mask_var, input_dim,
        num_layers, num_units, num_hyper_units, num_proj_units, 
        output_dim, grad_clipping, bidir=True, num_hyperlstm_layers=1):

    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)

    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
        hyper_layer = layer_idx <= num_hyperlstm_layers
        
        prev_fwd_layer = get_layer(layer_name, hyper_layer, prev_input_layer,
                            num_units, num_hyper_units, num_proj_units,
                            mask_layer, backwards=False, grad_clipping=grad_clipping)

     
        if bidir:
            prev_bwd_layer = get_layer(layer_name, hyper_layer, prev_input_layer,
                            num_units, num_hyper_units, num_proj_units,
                            mask_layer, backwards=True, grad_clipping=grad_clipping)
          
            prev_input_layer = ConcatLayer(incomings=[prev_fwd_layer, prev_bwd_layer],
                                   axis=-1)
            

        else:
            prev_input_layer = prev_fwd_layer

  
    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim)


