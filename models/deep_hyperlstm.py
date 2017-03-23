from lasagne import nonlinearities
from lasagne.layers import InputLayer, ConcatLayer, LSTMLayer
from libs.lasagne_libs.hyper_lhuc_layers import HyperLSTMLayer, \
        HyperLHUCLSTMLayer, SummarizingLHUCLSTMLayer, IVectorLHUCLSTMLayer, \
        SeqSumLHUCLSTMLayer

from models.utils import build_input_layer, build_ivector_layer, concatenate_layers, \
            build_sequence_dense_layer, build_sequence_summarizing_layer

def get_layer(layer_name, is_hyper_layer, prev_input_layer, 
        num_units, num_hyper_units, num_proj_units, 
        mask_layer, backwards, grad_clipping, speaker_layer=None, reparam='2sigmoid', 
        use_layer_norm=False,
        num_pred_layers=1, num_pred_units=100,
        pred_act='tanh'):

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
                                num_units, num_pred_layers, num_pred_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping,
                                reparam=reparam, use_layer_norm=use_layer_norm, pred_act=pred_act)
            

    elif layer_name == 'IVectorLHUCLSTMLayer':
        return IVectorLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_layers, num_pred_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping,
                                reparam=reparam, use_layer_norm=use_layer_norm, pred_act=pred_act)

    elif layer_name == 'SeqSumLHUCLSTMLayer':
        return SeqSumLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_layers, num_pred_units,
                                mask_input=mask_layer, backwards=backwards, grad_clipping=grad_clipping,
                                reparam=reparam, use_layer_norm=use_layer_norm, pred_act=pred_act)

    else:
        raise ValueError

def build_deep_hyper_lstm(layer_name, input_var, mask_var, input_dim,
        num_layers, num_units, num_hyper_units, num_proj_units, 
        output_dim, grad_clipping, bidir=True,  
        use_ivector_input=False, ivector_var=None, ivector_dim=100, 
        reparam='2sigmoid', use_layer_norm=False,
        num_pred_layers=1, num_pred_units=100, pred_act='tanh',
        num_seqsum_nodes=512, num_seqsum_layers=2, seqsum_output_dim=100):

    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)
  
    speaker_layer = None
    if ivector_var:
        speaker_layer = build_ivector_layer(ivector_dim, ivector_var)
    
    if use_ivector_input:
       input_layer = concatenate_layers(input_layer, speaker_layer)

    if layer_name == 'SeqSumLHUCLSTMLayer':

        speaker_layer = build_sequence_summarizing_layer(input_var, input_layer, 
                num_nodes=num_seqsum_nodes, num_layers=num_seqsum_layers, output_dim=seqsum_output_dim)
        
        

    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
 #       is_hyper_layer = layer_idx <= num_hyperlstm_layers
        is_hyper_layer = True
        
        prev_fwd_layer = get_layer(layer_name, is_hyper_layer, prev_input_layer,
                            num_units, num_hyper_units, num_proj_units,
                            mask_layer, backwards=False, grad_clipping=grad_clipping, speaker_layer=speaker_layer, 
                            reparam=reparam, use_layer_norm=use_layer_norm, num_pred_layers=num_pred_layers, 
                            num_pred_units=num_pred_units, pred_act=pred_act)

     
        if bidir:
            prev_bwd_layer = get_layer(layer_name, is_hyper_layer, prev_input_layer,
                            num_units, num_hyper_units, num_proj_units,
                            mask_layer, backwards=True, grad_clipping=grad_clipping, speaker_layer=speaker_layer,
                            reparam=reparam, use_layer_norm=use_layer_norm,
                            num_pred_layers=num_pred_layers, num_pred_units=num_pred_units, pred_act=pred_act)
          
            prev_input_layer = ConcatLayer(incomings=[prev_fwd_layer, prev_bwd_layer],
                                   axis=-1)
            

        else:
            prev_input_layer = prev_fwd_layer

  
    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim), speaker_layer


