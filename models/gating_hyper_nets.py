from theano import tensor as T
from lasagne import nonlinearities, init
from libs.lasagne_libs.layers import SequenceDenseLayer
from lasagne.layers import InputLayer, DenseLayer, ConcatLayer, SliceLayer, GlobalPoolLayer, DropoutLayer
from libs.lasagne_libs.hyper_layers import ScalingHyperLSTMLayer, ProjectionHyperLSTMLayer
from libs.lasagne_libs.recurrent_layers import (ProjectLSTMLayer,
                                                LayerNormProjectLSTMLayer,
                                                CondLayerNormProjectLSTMLayer,
                                                InputCondLayerNormProjectLSTMLayer,
                                                MaxCondLayerNormProjectLSTMLayer,
                                                MaxInputCondLayerNormProjectLSTMLayer)

def deep_scaling_hyper_model(input_var,
                             mask_var,
                             num_inputs,
                             num_outputs,
                             num_inner_units_list,
                             num_outer_units_list,
                             use_peepholes=False,
                             use_layer_norm=False,
                             learn_init=False,
                             grad_clipping=1.0,
                             get_inner_hid=False,
                             use_softmax=True):
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
    for num_inner_units, num_outer_units  in zip(num_inner_units_list, num_outer_units_list):
        # forward
        prev_fwd_layer = ScalingHyperLSTMLayer(inner_incoming=prev_input_layer,
                                               outer_incoming=prev_input_layer,
                                               mask_incoming=mask_layer,
                                               num_inner_units=num_inner_units,
                                               num_outer_units=num_outer_units,
                                               use_peepholes=use_peepholes,
                                               use_layer_norm=use_layer_norm,
                                               learn_init=learn_init,
                                               grad_clipping=grad_clipping,
                                               backwards=False,
                                               only_return_outer=False)
        # inner loop hidden
        prev_fwd_inner_layer = SliceLayer(incoming=prev_fwd_layer,
                                          indices=slice(0, num_inner_units),
                                          axis=-1)
        # outer loop hidden
        prev_fwd_outer_layer = SliceLayer(incoming=prev_fwd_layer,
                                          indices=slice(num_inner_units, None),
                                          axis=-1)

        # backward
        prev_bwd_layer = ScalingHyperLSTMLayer(inner_incoming=prev_input_layer,
                                               outer_incoming=prev_input_layer,
                                               mask_incoming=mask_layer,
                                               num_inner_units=num_inner_units,
                                               num_outer_units=num_outer_units,
                                               use_peepholes=use_peepholes,
                                               use_layer_norm=use_layer_norm,
                                               learn_init=learn_init,
                                               grad_clipping=grad_clipping,
                                               backwards=True,
                                               only_return_outer=False)
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
    print(prev_input_layer.output_shape)
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      mask_input=mask_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=nonlinearities.softmax if use_softmax else None)
    if get_inner_hid:
        return inner_hid_layer_list + [output_layer,]
    else:
        return output_layer


def deep_projection_hyper_model(input_var,
                                mask_var,
                                num_inputs,
                                num_outputs,
                                num_layers,
                                num_factors,
                                num_units,
                                grad_clipping=1):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #####################
    # stacked rnn layer #
    #####################
    inner_layer_list = []
    prev_input_layer = input_layer
    for l  in range(num_layers):
        # for first layer
        if l==0:
            # forward
            fwd_feat_layer = ProjectionHyperLSTMLayer(input_data_layer=prev_input_layer,
                                                      input_mask_layer=mask_layer,
                                                      num_factors=num_factors,
                                                      num_units=num_units,
                                                      grad_clipping=grad_clipping,
                                                      backwards=False)
            fwd_inner_layer = SliceLayer(incoming=fwd_feat_layer,
                                         indices=slice(0, num_factors),
                                         axis=-1)
            fwd_outer_layer = SliceLayer(incoming=fwd_feat_layer,
                                         indices=slice(num_factors, None),
                                         axis=-1)

            # backward
            bwd_feat_layer = ProjectionHyperLSTMLayer(input_data_layer=prev_input_layer,
                                                      input_mask_layer=mask_layer,
                                                      num_factors=num_factors,
                                                      num_units=num_units,
                                                      grad_clipping=grad_clipping,
                                                      backwards=True)
            bwd_inner_layer = SliceLayer(incoming=bwd_feat_layer,
                                         indices=slice(0, num_factors),
                                         axis=-1)
            bwd_outer_layer = SliceLayer(incoming=bwd_feat_layer,
                                         indices=slice(num_factors, None),
                                         axis=-1)

            prev_input_layer = ConcatLayer(incomings=[fwd_outer_layer, bwd_outer_layer],
                                           axis=-1)

            inner_layer_list.append(fwd_inner_layer)
            inner_layer_list.append(bwd_inner_layer)
        else:
            # forward
            fwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=False)

            # backward
            bwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=True)

            prev_input_layer = ConcatLayer(incomings=[fwd_feat_layer, bwd_feat_layer],
                                           axis=-1)

    ################
    # output layer #
    ################
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=None)
    return [output_layer, ] + inner_layer_list

def deep_projection_lstm_model(input_var,
                               mask_var,
                               num_inputs,
                               num_outputs,
                               num_layers,
                               num_factors,
                               num_units,
                               grad_clipping=1,
                               dropout=0.2):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #####################
    # stacked rnn layer #
    #####################
    prev_input_layer = input_layer
    for l  in range(num_layers):
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout)

        fwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          num_factors=num_factors,
                                          grad_clipping=grad_clipping,
                                          backwards=False)

        # backward
        bwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                          mask_input=mask_layer,
                                          num_units=num_units,
                                          num_factors=num_factors,
                                          grad_clipping=grad_clipping,
                                          backwards=True)

        prev_input_layer = ConcatLayer(incomings=[fwd_feat_layer, bwd_feat_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                    p=dropout)
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=None)
    return output_layer

def deep_projection_ln_lstm_model(input_var,
                                  mask_var,
                                  num_inputs,
                                  num_outputs,
                                  num_layers,
                                  num_factors,
                                  num_units,
                                  grad_clipping=1,
                                  dropout=0.2):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #####################
    # stacked rnn layer #
    #####################
    prev_input_layer = input_layer
    for l  in range(num_layers):
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout)

        fwd_feat_layer = LayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                   mask_input=mask_layer,
                                                   num_units=num_units,
                                                   num_factors=num_factors,
                                                   grad_clipping=grad_clipping,
                                                   backwards=False)

        # backward
        bwd_feat_layer = LayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                   mask_input=mask_layer,
                                                   num_units=num_units,
                                                   num_factors=num_factors,
                                                   grad_clipping=grad_clipping,
                                                   backwards=True)

        prev_input_layer = ConcatLayer(incomings=[fwd_feat_layer, bwd_feat_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                    p=dropout)
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=None)
    return output_layer

def deep_projection_cond_ln_model(input_var,
                                  mask_var,
                                  num_inputs,
                                  num_outputs,
                                  num_layers,
                                  num_conds,
                                  num_factors,
                                  num_units,
                                  grad_clipping=1,
                                  dropout=0.2):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #####################
    # stacked rnn layer #
    #####################
    cond_layer_list = []
    prev_input_layer = input_layer
    for l  in range(num_layers):
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout)
        if l<num_conds:
            # forward
            fwd_feat_layer = CondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                           mask_input=mask_layer,
                                                           num_units=num_units,
                                                           num_factors=num_factors,
                                                           grad_clipping=grad_clipping,
                                                           backwards=False)

            # backward
            bwd_feat_layer = CondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                           mask_input=mask_layer,
                                                           num_units=num_units,
                                                           num_factors=num_factors,
                                                           grad_clipping=grad_clipping,
                                                           backwards=True)

            cond_layer_list.append(fwd_feat_layer)
            cond_layer_list.append(bwd_feat_layer)
        else:
            # forward
            fwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=False)

            # backward
            bwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=True)

        prev_input_layer = ConcatLayer(incomings=[fwd_feat_layer, bwd_feat_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                    p=dropout)
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=None)
    return output_layer, cond_layer_list

def deep_projection_ivector_ln_model(input_var,
                                     cond_var,
                                     mask_var,
                                     num_inputs,
                                     num_outputs,
                                     num_layers,
                                     num_conds,
                                     num_factors,
                                     num_units,
                                     grad_clipping=1,
                                     dropout=0.2):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)
    ivec_layer = InputLayer(shape=(None, None, 100),
                            input_var=cond_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #####################
    # stacked rnn layer #
    #####################
    cond_layer_list = []
    prev_input_layer = input_layer
    for l  in range(num_layers):
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout)
        if l<num_conds:
            # forward
            fwd_feat_layer = InputCondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                                cond_input=ivec_layer,
                                                                mask_input=mask_layer,
                                                                num_units=num_units,
                                                                num_factors=num_factors,
                                                                grad_clipping=grad_clipping,
                                                                backwards=False)

            # backward
            bwd_feat_layer = InputCondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                                cond_input=ivec_layer,
                                                                mask_input=mask_layer,
                                                                num_units=num_units,
                                                                num_factors=num_factors,
                                                                grad_clipping=grad_clipping,
                                                                backwards=True)

            cond_layer_list.append(fwd_feat_layer)
            cond_layer_list.append(bwd_feat_layer)
        else:
            # forward
            fwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=False)

            # backward
            bwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=True)

        prev_input_layer = ConcatLayer(incomings=[fwd_feat_layer, bwd_feat_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                    p=dropout)
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=None)
    return output_layer, cond_layer_list

def deep_projection_max_cond_ln_model(input_var,
                                      mask_var,
                                      num_inputs,
                                      num_outputs,
                                      num_layers,
                                      num_conds,
                                      num_factors,
                                      num_units,
                                      grad_clipping=1,
                                      dropout=0.2):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #####################
    # stacked rnn layer #
    #####################
    cond_layer_list = []
    prev_input_layer = input_layer
    for l  in range(num_layers):
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout)
        if l<num_conds:
            # forward
            fwd_feat_layer = MaxCondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                              mask_input=mask_layer,
                                                              num_units=num_units,
                                                              num_factors=num_factors,
                                                              grad_clipping=grad_clipping,
                                                              backwards=False)

            # backward
            bwd_feat_layer = MaxCondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                              mask_input=mask_layer,
                                                              num_units=num_units,
                                                              num_factors=num_factors,
                                                              grad_clipping=grad_clipping,
                                                              backwards=True)

            cond_layer_list.append(fwd_feat_layer)
            cond_layer_list.append(bwd_feat_layer)
        else:
            # forward
            fwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=False)

            # backward
            bwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=True)

        prev_input_layer = ConcatLayer(incomings=[fwd_feat_layer, bwd_feat_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                    p=dropout)
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=None)
    return output_layer, cond_layer_list

def deep_projection_max_ivector_ln_model(input_var,
                                         cond_var,
                                         mask_var,
                                         num_inputs,
                                         num_outputs,
                                         num_layers,
                                         num_conds,
                                         num_factors,
                                         num_units,
                                         grad_clipping=1,
                                         dropout=0.2):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)
    ivec_layer = InputLayer(shape=(None, None, 100),
                            input_var=cond_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    #####################
    # stacked rnn layer #
    #####################
    cond_layer_list = []
    prev_input_layer = input_layer
    for l  in range(num_layers):
        prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                        p=dropout)
        if l<num_conds:
            # forward
            fwd_feat_layer = MaxInputCondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                                   cond_input=ivec_layer,
                                                                   mask_input=mask_layer,
                                                                   num_units=num_units,
                                                                   num_factors=num_factors,
                                                                   grad_clipping=grad_clipping,
                                                                   backwards=False)

            # backward
            bwd_feat_layer = MaxInputCondLayerNormProjectLSTMLayer(incoming=prev_input_layer,
                                                                   cond_input=ivec_layer,
                                                                   mask_input=mask_layer,
                                                                   num_units=num_units,
                                                                   num_factors=num_factors,
                                                                   grad_clipping=grad_clipping,
                                                                   backwards=True)

            cond_layer_list.append(fwd_feat_layer)
            cond_layer_list.append(bwd_feat_layer)
        else:
            # forward
            fwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=False)

            # backward
            bwd_feat_layer = ProjectLSTMLayer(incoming=prev_input_layer,
                                              mask_input=mask_layer,
                                              num_units=num_units,
                                              num_factors=num_factors,
                                              grad_clipping=grad_clipping,
                                              backwards=True)

        prev_input_layer = ConcatLayer(incomings=[fwd_feat_layer, bwd_feat_layer],
                                       axis=-1)

    ################
    # output layer #
    ################
    prev_input_layer = DropoutLayer(incoming=prev_input_layer,
                                    p=dropout)
    output_layer = SequenceDenseLayer(incoming=prev_input_layer,
                                      num_outputs=num_outputs,
                                      nonlinearity=None)
    return output_layer, cond_layer_list