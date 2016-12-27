from lasagne.layers import MergeLayer
from lasagne.init import Constant
import theano.tensor as T
import theano.tensor.nnet.nnet as nnet 


def two_sigmoid(x):
    return 2 * nnet.sigmoid(x)

def exp(x):
    return T.exp(x)

class LHUCLayer(MergeLayer):
    def __init__(self,
                 incoming, speaker_input_layer, num_speakers, psi,
                 W=Constant(),
                 **kwargs):

        incomings = [incoming, speaker_input_layer]
        super(LHUCLayer, self).__init__(incomings, **kwargs)

        m_batch, n_time_steps, n_features = self.input_shapes[0]
        self.num_speakers = num_speakers
        self.num_units = n_features 
        self.W = self.add_param(W, (self.num_speakers, self.num_units), name='W_LHUC', speaker_dependent=True)
        self.psi = psi
    
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        prev_layer_input, speaker_input = inputs
        scaled_W = self.psi(self.W[speaker_input])

        return prev_layer_input*scaled_W[:, None, :]
