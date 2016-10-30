from libs.lasagne.layers import SequenceSoftmaxLayer
from lasagne.layers import InputLayer, DenseLayer
from libs.lasagne.blocks import BidirLSTMBlock

def deep_bidir_lstm_model(input_var,
                          mask_var,
                          num_inputs,
                          num_units_list,
                          num_outputs,
                          dropout_ratio=0.2,
                          use_layer_norm=True,
                          learn_init=False,
                          grad_clipping=0.0):
    ###############
    # input layer #
    ###############
    input_layer = InputLayer(shape=(None, None, num_inputs),
                             input_var=input_var)

    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)

    ############################
    # stacked bidir lstm layer #
    ############################
    num_layers = len(num_units_list)
    prev_input_layer = input_layer
    for l in range(num_layers):
        prev_input_layer = BidirLSTMBlock(data_layer=prev_input_layer,
                                          mask_layer=mask_layer,
                                          num_units=num_units_list[l],
                                          dropout_ratio=dropout_ratio,
                                          use_layer_norm=use_layer_norm,
                                          learn_init=learn_init,
                                          grad_clipping=grad_clipping)

    ################
    # output layer #
    ################
    output_layer = SequenceSoftmaxLayer(incoming=DenseLayer(incoming=prev_input_layer,
                                                            num_units=num_outputs,
                                                            nonlinearity=None),
                                        mask_input=mask_layer)


    return output_layer





