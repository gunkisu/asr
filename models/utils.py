from lasagne.layers import InputLayer, ConcatLayer

def build_input_layer(input_dim, input_var, mask_var):
    input_layer = InputLayer(shape=(None, None, input_dim),
                             input_var=input_var)
    mask_layer = InputLayer(shape=(None, None),
                            input_var=mask_var)


    return (input_layer, mask_layer)


def build_ivector_layer(ivector_dim, ivector_var):
    ivector_layer = InputLayer(shape=(None, None, ivector_dim,), input_var=ivector_var)
    return ivector_layer
  
def concatenate_layers(layer1, layer2):
    return ConcatLayer(incomings=[layer1, layer2], axis=-1)
