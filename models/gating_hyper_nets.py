from lasagne import nonlinearities
from libs.lasagne.layers import SequenceDenseLayer
from lasagne.layers import InputLayer, DropoutLayer, ConcatLayer, SliceLayer
from libs.lasagne.hyper_layers import GatingHyperLSTMLayer

def deep_gating_hyper_model(input_var,
                            mask_var,
                            num_inputs,
                            num_inner_units_list,
                            num_factor_units_list,
                            num_outer_units_list,
                            num_outputs,
                            dropout_ratio=0.2,
                            use_layer_norm=True,
                            gating_nonlinearity=None,
                            weight_noise=0.0,
                            peepholes=False,
                            learn_init=False,
                            get_inner_hid=False,
                            grad_clipping=1.0,
                            use_softmax=True,):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ####################################
    # stacked bidir gating hyper layer #
    ####################################
    inner_hid_layer_list = []
    prev_input_layer = input_layer
    for num_inner_units, num_factor_units, num_outer_units  in zip(num_inner_units_list, num_factor_units_list, num_outer_units_list):
        # drop out
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout_ratio)

        # forward
        prev_fwd_layer = GatingHyperLSTMLayer(inner_incoming=prev_input_layer,
                                              outer_incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_inner_units=num_inner_units,
                                              num_factor_units=num_factor_units,
                                              num_outer_units=num_outer_units,
                                              gating_nonlinearity=gating_nonlinearity,
                                              dropout_ratio=dropout_ratio,
                                              use_layer_norm=use_layer_norm,
                                              weight_noise=weight_noise,
                                              peepholes=peepholes,
                                              learn_init=learn_init,
                                              grad_clipping=grad_clipping,
                                              backwards=False,
                                              output_inner_hid=True)

        # inner loop hidden
        prev_fwd_inner_layer = SliceLayer(incoming=prev_fwd_layer,
                                          indices=slice(0, num_inner_units),
                                          axis=-1)
        # outer loop hidden
        prev_fwd_outer_layer = SliceLayer(incoming=prev_fwd_layer,
                                          indices=slice(num_inner_units, None),
                                          axis=-1)

        # backward
        prev_bwd_layer = GatingHyperLSTMLayer(inner_incoming=prev_input_layer,
                                              outer_incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_inner_units=num_inner_units,
                                              num_factor_units=num_factor_units,
                                              num_outer_units=num_outer_units,
                                              gating_nonlinearity=gating_nonlinearity,
                                              dropout_ratio=dropout_ratio,
                                              use_layer_norm=use_layer_norm,
                                              weight_noise=weight_noise,
                                              peepholes=peepholes,
                                              learn_init=learn_init,
                                              grad_clipping=grad_clipping,
                                              backwards=True,
                                              output_inner_hid=True)
        # inner loop hidden
        prev_bwd_inner_layer = SliceLayer(incoming=prev_bwd_layer,
                                          indices=slice(0, num_inner_units),
                                          axis=-1)

        # outer loop hidden
        prev_bwd_outer_layer = SliceLayer(incoming=prev_bwd_layer,
                                          indices=slice(num_inner_units, None),
                                          axis=-1)

        # concatenate bidirectional
        prev_input_layer = ConcatLayer(incomings=[prev_fwd_outer_layer, prev_bwd_outer_layer],
                                       axis=-1)
        # get inner loop hiddens
        inner_hid_layer_list.append(prev_fwd_inner_layer)
        inner_hid_layer_list.append(prev_bwd_inner_layer)


    ################
    # output layer #
    ################
    output_layer = DropoutLayer(incoming=prev_input_layer,
                                p=dropout_ratio)
    output_layer = SequenceDenseLayer(incoming=output_layer,
                                      num_outputs=num_outputs,
                                      mask_input=mask_layer,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    if get_inner_hid:
        return inner_hid_layer_list + [output_layer,]
    else:
        return output_layer

def deep_gating_skip_hyper_model(input_var,
                                 mask_var,
                                 num_inputs,
                                 num_inner_units_list,
                                 num_factor_units_list,
                                 num_outer_units_list,
                                 num_outputs,
                                 dropout_ratio=0.2,
                                 use_layer_norm=True,
                                 gating_nonlinearity=None,
                                 weight_noise=0.0,
                                 peepholes=False,
                                 learn_init=False,
                                 get_inner_hid=False,
                                 grad_clipping=1.0,
                                 use_softmax=True,):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ####################################
    # stacked bidir gating hyper layer #
    ####################################
    inner_hid_layer_list = []
    prev_input_layer = input_layer
    for num_inner_units, num_factor_units, num_outer_units  in zip(num_inner_units_list, num_factor_units_list, num_outer_units_list):
        # drop out
        skip_input_layer = DropoutLayer(incoming=input_layer,
                                        p=dropout_ratio)
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout_ratio)

        # forward
        prev_fwd_layer = GatingHyperLSTMLayer(inner_incoming=skip_input_layer,
                                              outer_incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_inner_units=num_inner_units,
                                              num_factor_units=num_factor_units,
                                              num_outer_units=num_outer_units,
                                              gating_nonlinearity=gating_nonlinearity,
                                              dropout_ratio=dropout_ratio,
                                              use_layer_norm=use_layer_norm,
                                              weight_noise=weight_noise,
                                              peepholes=peepholes,
                                              learn_init=learn_init,
                                              grad_clipping=grad_clipping,
                                              backwards=False,
                                              output_inner_hid=True)

        # inner loop hidden
        prev_fwd_inner_layer = SliceLayer(incoming=prev_fwd_layer,
                                          indices=slice(0, num_inner_units),
                                          axis=-1)
        # outer loop hidden
        prev_fwd_outer_layer = SliceLayer(incoming=prev_fwd_layer,
                                          indices=slice(num_inner_units, None),
                                          axis=-1)

        # backward
        prev_bwd_layer = GatingHyperLSTMLayer(inner_incoming=skip_input_layer,
                                              outer_incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_inner_units=num_inner_units,
                                              num_factor_units=num_factor_units,
                                              num_outer_units=num_outer_units,
                                              gating_nonlinearity=gating_nonlinearity,
                                              dropout_ratio=dropout_ratio,
                                              use_layer_norm=use_layer_norm,
                                              weight_noise=weight_noise,
                                              peepholes=peepholes,
                                              learn_init=learn_init,
                                              grad_clipping=grad_clipping,
                                              backwards=True,
                                              output_inner_hid=True)
        # inner loop hidden
        prev_bwd_inner_layer = SliceLayer(incoming=prev_bwd_layer,
                                          indices=slice(0, num_inner_units),
                                          axis=-1)

        # outer loop hidden
        prev_bwd_outer_layer = SliceLayer(incoming=prev_bwd_layer,
                                          indices=slice(num_inner_units, None),
                                          axis=-1)

        # concatenate bidirectional
        prev_input_layer = ConcatLayer(incomings=[prev_fwd_outer_layer, prev_bwd_outer_layer],
                                       axis=-1)
        # get inner loop hiddens
        inner_hid_layer_list.append(prev_fwd_inner_layer)
        inner_hid_layer_list.append(prev_bwd_inner_layer)


    ################
    # output layer #
    ################
    output_layer = DropoutLayer(incoming=prev_input_layer,
                                p=dropout_ratio)
    output_layer = SequenceDenseLayer(incoming=output_layer,
                                      num_outputs=num_outputs,
                                      mask_input=mask_layer,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    if get_inner_hid:
        return inner_hid_layer_list + [output_layer,]
    else:
        return output_layer