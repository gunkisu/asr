from lasagne import nonlinearities
from lasagne.layers import InputLayer, ConcatLayer, LSTMLayer
from libs.lasagne_libs.hyper_lhuc_layers import HyperLSTMLayer, \
        HyperLHUCLSTMLayer, SummarizingLHUCLSTMLayer, IVectorLHUCLSTMLayer

from models.utils import build_input_layer, build_ivector_layer, concatenate_layers, \
            build_sequence_dense_layer

def get_layer(layer_name, is_hyper_layer, prev_input_layer, 
        num_units, num_hyper_units, num_proj_units, 
        mask_layer, backwards, grad_clipping, ivector_layer=None, reparam='2sigmoid', use_layer_norm=False):

    if not is_hyper_layer:
        return LSTMLayer(prev_input_layer, num_units,
                                   mask_input=mask_layer,
                                   grad_clipping=grad_clipping,
                                   backwards=backwards)
 
    if layer_name == 'HyperLSTMLayer':
        return HyperLSTMLayer(prev_input_layer,
                                num_units, num_hyper_units, num_proj_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping,
                                use_layer_norm=use_layer_norm)
    elif layer_name == 'HyperLHUCLSTMLayer':
        return HyperLHUCLSTMLayer(prev_input_layer,
                                num_units, num_hyper_units, num_proj_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping,
                                reparam=reparam, use_layer_norm=use_layer_norm)
    elif layer_name == 'SummarizingLHUCLSTMLayer': 
        return SummarizingLHUCLSTMLayer(prev_input_layer,
                                num_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping,
                                reparam=reparam, use_layer_norm=use_layer_norm)
            

    elif layer_name == 'IVectorLHUCLSTMLayer':
        return IVectorLHUCLSTMLayer(prev_input_layer, ivector_layer,
                                num_units, 
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping,
                                reparam=reparam, use_layer_norm=use_layer_norm)


def build_deep_hyper_lstm(layer_name, input_var, mask_var, input_dim,
        num_layers, num_units, num_hyper_units, num_proj_units, 
        output_dim, grad_clipping, bidir=True, num_hyperlstm_layers=1, 
        use_ivector_input=False, ivector_var=None, ivector_dim=100, reparam='2sigmoid', use_layer_norm=False):

    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)

    ivector_layer = None
    if ivector_var:
        ivector_layer = build_ivector_layer(ivector_dim, ivector_var)
    
    if use_ivector_input:
       input_layer = concatenate_layers(input_layer, ivector_layer)
  
    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
        is_hyper_layer = layer_idx <= num_hyperlstm_layers
        
        prev_fwd_layer = get_layer(layer_name, is_hyper_layer, prev_input_layer,
                            num_units, num_hyper_units, num_proj_units,
                            mask_layer, backwards=False, grad_clipping=grad_clipping, ivector_layer=ivector_layer, 
                            reparam=reparam, use_layer_norm=use_layer_norm)

     
        if bidir:
            prev_bwd_layer = get_layer(layer_name, is_hyper_layer, prev_input_layer,
                            num_units, num_hyper_units, num_proj_units,
                            mask_layer, backwards=True, grad_clipping=grad_clipping, ivector_layer=ivector_layer,
                            reparam=reparam, use_layer_norm=use_layer_norm)
          
            prev_input_layer = ConcatLayer(incomings=[prev_fwd_layer, prev_bwd_layer],
                                   axis=-1)
            

        else:
            prev_input_layer = prev_fwd_layer

  
    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim)


