from lasagne import nonlinearities
from libs.lasagne.layers import SequenceDenseLayer, SequenceLayerNormLayer
from lasagne.layers import InputLayer, DropoutLayer, ConcatLayer
from libs.lasagne.hyper_layers import BiDirScaleHyperLSTMLayer, ScaleHyperLSTMLayer

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
                                      grad_clipping=0.0):
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
        prev_input_layer = SequenceLayerNormLayer(incoming=input_layer)
    else:
        prev_input_layer = input_layer
    for l in range(num_layers):
        prev_fwd_input_layer = ScaleHyperLSTMLayer(incoming=DropoutLayer(prev_input_layer, p=dropout_ratio),
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
        prev_bwd_input_layer = ScaleHyperLSTMLayer(incoming=DropoutLayer(prev_input_layer, p=dropout_ratio),
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
        # prev_input_layer = BiDirScaleHyperLSTMLayer(incoming=DropoutLayer(prev_input_layer, p=dropout_ratio),
        #                                             mask_input=mask_layer,
        #                                             num_inner_units=num_inner_units_list[l],
        #                                             num_inner_factor_units=num_factor_units_list[l],
        #                                             num_outer_units=num_outer_units_list[l],
        #                                             dropout_ratio=dropout_ratio,
        #                                             use_layer_norm=use_layer_norm,
        #                                             weight_noise=weight_noise,
        #                                             learn_init=learn_init,
        #                                             grad_clipping=grad_clipping)


    ################
    # output layer #
    ################
    output_layer = SequenceDenseLayer(incoming=DropoutLayer(prev_input_layer, p=dropout_ratio),
                                      num_outputs=num_outputs,
                                      mask_input=mask_layer,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    return output_layer





