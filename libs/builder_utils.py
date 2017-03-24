from lasagne.layers import Layer, InputLayer, ConcatLayer, DenseLayer, reshape
from lasagne import nonlinearities
import theano.tensor as T

class SummarizingLayer(Layer):
    '''Summarizing the ourput of the previous layer by taking the mean or
    average over the time axis (axis=1). Assume that the shape of the output 
    of the previous is (n_batch, n_time_steps, n_feat). 
    ''' 

    def __init__(self, incoming, pool_function=T.mean, **kwargs):
        super(SummarizingLayer, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_output_for(self, input, **kwargs):
        # n_batch, n_time_steps, n_feat

        # n_batch, n_feat
        return self.pool_function(input, axis=1)

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

def build_sequence_summarizing_layer(input_var, input_layer, 
        num_nodes=512, num_layers=2, output_dim=100):
    '''Default arguments values from the 2016 ICASSP paper'''

    # input_layer: n_batch, n_time_steps, n_feat
    n_batch, n_time_steps, _ = input_var.shape
    
    # n_batch, n_feat
    prev_layer = reshape(input_layer, (-1, [2]))
    for i in range(num_layers):
        prev_layer = DenseLayer(prev_layer, 
                num_units=num_nodes, nonlinearity=nonlinearities.tanh)
    
    dense_layer= DenseLayer(prev_layer, num_units=output_dim, nonlinearity=nonlinearities.identity)
    
    return SummarizingLayer(reshape(dense_layer, (n_batch, n_time_steps, output_dim)))



