from lasagne.layers import InputLayer, ConcatLayer, DenseLayer, reshape
from lasagne import nonlinearities

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


def build_sequence_dense_layer(input_var, input_layer, output_dim):
    n_batch, n_time_steps, _ = input_var.shape
    dense_layer = DenseLayer(reshape(input_layer, (-1, [2])), 
            num_units=output_dim, nonlinearity=nonlinearities.softmax)
    return reshape(dense_layer, (n_batch, n_time_steps, output_dim))
