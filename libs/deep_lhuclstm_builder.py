from lasagne import nonlinearities
from lasagne.layers import InputLayer, ConcatLayer, LSTMLayer
from libs.lhuclstm_layers import IVectorLHUCLSTMLayer, SeqSumLHUCLSTMLayer

from libs.builder_utils import build_input_layer, build_ivector_layer, concatenate_layers, \
            build_sequence_dense_layer, build_sequence_summarizing_layer

def build_deep_lhuclstm_seqsum(layer_name, input_var, mask_var, input_dim,
        num_layers, num_units, num_proj_units, output_dim, num_pred_layers, num_pred_units, 
        num_seqsum_units, num_seqsum_layers, seqsum_output_dim, batch_size,
        grad_clipping, is_bidir, use_layer_norm, use_ivector_input, ivector_dim, ivector_var=None):

    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)

    speaker_layer = None
    if ivector_var:
        speaker_layer = build_ivector_layer(ivector_dim, ivector_var)
    if use_ivector_input:
        input_layer = concatenate_layers(input_layer, speaker_layer)

    speaker_layer = build_sequence_summarizing_layer(input_var, input_layer, 
       num_units=num_seqsum_units, num_layers=num_seqsum_layers, output_dim=seqsum_output_dim)
  
    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
        prev_fwd_layer = SeqSumLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_units, num_pred_layers, batch_size,
                                backwards=False, grad_clipping=grad_clipping, mask_input=mask_layer,
                                num_proj_units=num_proj_units, use_layer_norm=use_layer_norm)
        
        if is_bidir:
            prev_bwd_layer = SeqSumLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_units, num_pred_layers, batch_size,
                                backwards=True, grad_clipping=grad_clipping, mask_input=mask_layer,
                                num_proj_units=num_proj_units, use_layer_norm=use_layer_norm)        

            prev_input_layer = ConcatLayer(incomings=[prev_fwd_layer, prev_bwd_layer],
                                   axis=-1)
            

        else:
            prev_input_layer = prev_fwd_layer

  
    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim), speaker_layer



def build_deep_lhuclstm_ivector(layer_name, input_var, mask_var, input_dim,
        num_layers, num_units, num_proj_units, output_dim, num_pred_layers, num_pred_units, 
        num_seqsum_units, num_seqsum_layers, seqsum_output_dim, batch_size,
        grad_clipping, is_bidir, use_layer_norm, use_ivector_input, ivector_dim, ivector_var):

    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)
  
    speaker_layer = None
    if ivector_var:
        speaker_layer = build_ivector_layer(ivector_dim, ivector_var)
    
    if use_ivector_input:
        input_layer = concatenate_layers(input_layer, speaker_layer)

    prev_input_layer = input_layer
    for layer_idx in range(1, num_layers+1):
        prev_fwd_layer = IVectorLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_units, num_pred_layers, batch_size,
                                backwards=False, grad_clipping=grad_clipping, mask_input=mask_layer,
                                num_proj_units=num_proj_units, use_layer_norm=use_layer_norm)
        
        if is_bidir:
            prev_bwd_layer = IVectorLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_units, num_pred_layers, batch_size,
                                backwards=True, grad_clipping=grad_clipping, mask_input=mask_layer,
                                num_proj_units=num_proj_units, use_layer_norm=use_layer_norm)        

            prev_input_layer = ConcatLayer(incomings=[prev_fwd_layer, prev_bwd_layer],
                                   axis=-1)
            

        else:
            prev_input_layer = prev_fwd_layer
  
    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim), speaker_layer

def build_deep_uni_lhuclstm_seqsum_tbptt(layer_name, input_var, mask_var, input_dim,
        num_layers, num_units, num_proj_units, output_dim, num_pred_layers, num_pred_units, 
        num_seqsum_units, num_seqsum_layers, seqsum_output_dim, batch_size,
        grad_clipping, use_layer_norm, use_ivector_input, ivector_dim, ivector_var=None):

    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)

    speaker_layer = None
    if ivector_var:
        speaker_layer = build_ivector_layer(ivector_dim, ivector_var)
    if use_ivector_input:
        input_layer = concatenate_layers(input_layer, speaker_layer)

    speaker_layer = build_sequence_summarizing_layer(input_var, input_layer, 
       num_units=num_seqsum_units, num_layers=num_seqsum_layers, output_dim=seqsum_output_dim)
  
    prev_input_layer = input_layer
    tbptt_layers = []
    for layer_idx in range(1, num_layers+1):
        prev_fwd_layer = SeqSumLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_units, num_pred_layers, batch_size,
                                backwards=False, grad_clipping=grad_clipping, mask_input=mask_layer,
                                num_proj_units=num_proj_units, use_layer_norm=use_layer_norm)
        
        prev_input_layer = prev_fwd_layer

        tbptt_layers.append(prev_fwd_layer)

  
    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim), speaker_layer, tbptt_layers



def build_deep_uni_lhuclstm_ivector_tbptt(layer_name, input_var, mask_var, input_dim,
        num_layers, num_units, num_proj_units, output_dim, num_pred_layers, num_pred_units, 
        num_seqsum_units, num_seqsum_layers, seqsum_output_dim, batch_size,
        grad_clipping, use_layer_norm, use_ivector_input, ivector_dim, ivector_var):

    input_layer, mask_layer = build_input_layer(input_dim, input_var, mask_var)
  
    speaker_layer = None
    if ivector_var:
        speaker_layer = build_ivector_layer(ivector_dim, ivector_var)
    
    if use_ivector_input:
        input_layer = concatenate_layers(input_layer, speaker_layer)

    prev_input_layer = input_layer
    tbptt_layers = []
    for layer_idx in range(1, num_layers+1):
        prev_fwd_layer = IVectorLHUCLSTMLayer(prev_input_layer, speaker_layer,
                                num_units, num_pred_units, num_pred_layers, batch_size,
                                backwards=False, grad_clipping=grad_clipping, mask_input=mask_layer,
                                num_proj_units=num_proj_units, use_layer_norm=use_layer_norm)
        
        prev_input_layer = prev_fwd_layer
        tbptt_layers.append(prev_fwd_layer)
  
    return build_sequence_dense_layer(input_var, prev_input_layer, output_dim), speaker_layer, tbptt_layers