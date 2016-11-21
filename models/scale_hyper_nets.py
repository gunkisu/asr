from lasagne import nonlinearities
from libs.lasagne.layers import SequenceDenseLayer, SequenceLayerNormLayer
from lasagne.layers import InputLayer, DropoutLayer, ConcatLayer, SliceLayer
from libs.lasagne.hyper_layers import ScaleHyperLSTMLayer

def deep_bidir_scale_hyper_lstm_model(input_var,
                                      mask_var,
                                      num_inputs,
                                      num_inner_units_list,
                                      num_factor_units_list,
                                      num_outer_units_list,
                                      num_outputs,
                                      dropout_ratio=0.2,
                                      use_layer_norm=True,
                                      use_input_layer_norm=False,
                                      weight_noise=0.0,
                                      use_softmax=True,
                                      learn_init=False,
                                      grad_clipping=1.0,
                                      get_inner_hid=False):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ###################################
    # stacked bidir scale hyper layer #
    ###################################
    inner_hid_layer_list = []

    num_layers = len(num_inner_units_list)
    if use_input_layer_norm:
        prev_input_layer = SequenceLayerNormLayer(incoming=input_layer)
    else:
        prev_input_layer = input_layer

    prev_input_layer = DropoutLayer(incoming=prev_input_layer, p=dropout_ratio)
    for l in range(num_layers):
        prev_fwd_input_layer = ScaleHyperLSTMLayer(inner_incoming=prev_input_layer,
                                                   outer_incoming=prev_input_layer,
                                                   mask_input=mask_layer,
                                                   num_inner_units=num_inner_units_list[l],
                                                   num_factor_units=num_factor_units_list[l],
                                                   num_outer_units=num_outer_units_list[l],
                                                   dropout_ratio=dropout_ratio,
                                                   use_layer_norm=use_layer_norm,
                                                   weight_noise=weight_noise,
                                                   learn_init=learn_init,
                                                   grad_clipping=grad_clipping,
                                                   backwards=False,
                                                   output_inner_hid=True)

        prev_fwd_inner_layer = SliceLayer(incoming=prev_fwd_input_layer,
                                          indices=slice(0, num_inner_units_list[l]),
                                          axis=-1)
        prev_fwd_outer_layer = SliceLayer(incoming=prev_fwd_input_layer,
                                          indices=slice(num_inner_units_list[l], None),
                                          axis=-1)

        prev_bwd_input_layer = ScaleHyperLSTMLayer(inner_incoming=prev_input_layer,
                                                   outer_incoming=prev_input_layer,
                                                   mask_input=mask_layer,
                                                   num_inner_units=num_inner_units_list[l],
                                                   num_factor_units=num_factor_units_list[l],
                                                   num_outer_units=num_outer_units_list[l],
                                                   dropout_ratio=dropout_ratio,
                                                   use_layer_norm=use_layer_norm,
                                                   weight_noise=weight_noise,
                                                   learn_init=learn_init,
                                                   grad_clipping=grad_clipping,
                                                   backwards=True,
                                                   output_inner_hid=True)

        prev_bwd_inner_layer = SliceLayer(incoming=prev_bwd_input_layer,
                                          indices=slice(0, num_inner_units_list[l]),
                                          axis=-1)
        prev_bwd_outer_layer = SliceLayer(incoming=prev_bwd_input_layer,
                                          indices=slice(num_inner_units_list[l], None),
                                          axis=-1)

        prev_input_layer = ConcatLayer([prev_fwd_outer_layer, prev_bwd_outer_layer], axis=-1)
        prev_input_layer = DropoutLayer(incoming=prev_input_layer, p=dropout_ratio)

        inner_hid_layer_list.append(prev_fwd_inner_layer)
        inner_hid_layer_list.append(prev_bwd_inner_layer)


    ################
    # output layer #
    ################
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      mask_input=mask_layer,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    if get_inner_hid:
        return inner_hid_layer_list + [output_layer,]
    else:
        return output_layer

def deep_bidir_scale_hyper_lstm_skip_model(input_var,
                                           mask_var,
                                           num_inputs,
                                           num_inner_units_list,
                                           num_factor_units_list,
                                           num_outer_units_list,
                                           num_outputs,
                                           dropout_ratio=0.2,
                                           use_layer_norm=True,
                                           use_input_layer_norm=False,
                                           weight_noise=0.0,
                                           use_softmax=True,
                                           learn_init=False,
                                           grad_clipping=1.0):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ###################################
    # stacked bidir scale hyper layer #
    ###################################
    num_layers = len(num_inner_units_list)
    if use_input_layer_norm:
        input_layer = SequenceLayerNormLayer(incoming=input_layer)
    else:
        input_layer = input_layer

    prev_input_layer = DropoutLayer(incoming=input_layer, p=dropout_ratio)
    for l in range(num_layers):
        input_layer = DropoutLayer(incoming=input_layer, p=dropout_ratio),

        prev_fwd_input_layer = ScaleHyperLSTMLayer(inner_incoming=input_layer,
                                                   outer_incoming=prev_input_layer,
                                                   mask_input=mask_layer,
                                                   num_inner_units=num_inner_units_list[l],
                                                   num_inner_factor_units=num_factor_units_list[l],
                                                   num_outer_units=num_outer_units_list[l],
                                                   dropout_ratio=dropout_ratio,
                                                   use_layer_norm=use_layer_norm,
                                                   weight_noise=weight_noise,
                                                   learn_init=learn_init,
                                                   grad_clipping=grad_clipping,
                                                   backwards=False)
        prev_bwd_input_layer = ScaleHyperLSTMLayer(inner_incoming=input_layer,
                                                   outer_incoming=prev_input_layer,
                                                   mask_input=mask_layer,
                                                   num_inner_units=num_inner_units_list[l],
                                                   num_inner_factor_units=num_factor_units_list[l],
                                                   num_outer_units=num_outer_units_list[l],
                                                   dropout_ratio=dropout_ratio,
                                                   use_layer_norm=use_layer_norm,
                                                   weight_noise=weight_noise,
                                                   learn_init=learn_init,
                                                   grad_clipping=grad_clipping,
                                                   backwards=True)
        prev_input_layer = ConcatLayer([prev_fwd_input_layer, prev_bwd_input_layer], axis=-1)
        prev_input_layer = DropoutLayer(incoming=prev_input_layer, p=dropout_ratio)


    ################
    # output layer #
    ################
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      mask_input=mask_layer,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    return output_layer





