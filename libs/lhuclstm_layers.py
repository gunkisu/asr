import theano
import numpy
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import relu
from lasagne import init, nonlinearities
from lasagne.layers import Layer, MergeLayer, Gate

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

def ln(input, alpha, beta=None):
    output = (input - T.mean(input, axis=1, keepdims=True)) / T.sqrt(T.var(input, axis=1, keepdims=True) + eps)
    output *= alpha[None, :]
    if beta:
        output += beta[None, :]
    return output

from abc import ABCMeta, abstractmethod, abstractproperty

class LSTMOpMixin(object):
    def slice_w(self, x, n):
        s = x[:, n*self.num_units:(n+1)*self.num_units]
        if self.num_units == 1:
            s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
        return s

    def add_gate_params(self, gate_name):
        num_prev_units = self.num_proj_units if self.num_proj_units else self.num_units

        return (self.add_param(init.Orthogonal(), (num_prev_units, self.num_units),
                               name="W_h_{}".format(gate_name)),
                self.add_param(init.Orthogonal(), (self.num_inputs, self.num_units),
                               name="W_x_{}".format(gate_name)),
                self.add_param(init.Constant(0.0), (self.num_units,),
                               name="b_{}".format(gate_name),
                               regularizable=False))

    def init_main_lstm_weights(self):
        (self.W_h_ig, self.W_x_ig, self.b_ig) = self.add_gate_params('ig')
        (self.W_h_fg, self.W_x_fg, self.b_fg) = self.add_gate_params('fg')
        (self.W_h_c, self.W_x_c, self.b_c) = self.add_gate_params('c')
        (self.W_h_og, self.W_x_og, self.b_og) = self.add_gate_params('og')

        if self.num_proj_units:
            self.W_p = self.add_param(init.Orthogonal(), (self.num_units, self.num_proj_units), name="W_p")

        self.cell_init = self.add_param(init.Constant(0.0), (1, self.num_units), name="cell_init",
            trainable=False, regularizable=False)

        if self.num_proj_units:
            self.hid_init = self.add_param(init.Constant(0.0), (1, self.num_proj_units), name="hid_init",
                trainable=False, regularizable=False)
        else:
            self.hid_init = self.add_param(init.Constant(0.0), (1, self.num_units), name="hid_init",
                trainable=False, regularizable=False)

        if self.use_layer_norm:
            self.W_x_alpha = self.add_param(spec=init.Constant(1.0), shape=(self.num_units*4,), name="W_x_alpha")
            self.W_h_alpha = self.add_param(spec=init.Constant(1.0), shape=(self.num_units*4,), name="W_h_alpha")
            self.W_c_alpha = self.add_param(spec=init.Constant(1.0), shape=(self.num_units,), name="W_c_alpha")
            self.W_c_beta = self.add_param(spec=init.Constant(0.0), shape=(self.num_units,), name="W_c_beta", regularizable=False)

    def step(self, input_n, cell_previous, hid_previous, *args):
        # Precomputed: input = T.dot(input, self.W_x_stacked)

        if self.use_layer_norm:
            gates = ln(input_n, self.W_x_alpha)
            gates += ln(T.dot(hid_previous, self.W_h_stacked), self.W_h_alpha)
        else:
            gates = input_n 
            gates += T.dot(hid_previous, self.W_h_stacked)
        gates += self.b_stacked
   
        ingate, forgetgate, cell_input, outgate = \
            [self.slice_w(gates, i) for i in range(4)]

        if self.grad_clipping:
            ingate = theano.gradient.grad_clip(ingate, -self.grad_clipping, self.grad_clipping)
            forgetgate = theano.gradient.grad_clip(forgetgate, -self.grad_clipping, self.grad_clipping)
            cell_input = theano.gradient.grad_clip(cell_input, -self.grad_clipping, self.grad_clipping)
            outgate = theano.gradient.grad_clip(outgate, -self.grad_clipping, self.grad_clipping)

        ingate = T.nnet.sigmoid(ingate)
        forgetgate = T.nnet.sigmoid(forgetgate)
        cell_input = T.tanh(cell_input)
        outgate = T.nnet.sigmoid(outgate)

        cell = forgetgate*cell_previous + ingate*cell_input
        
        if self.use_layer_norm:
            outcell = ln(cell, self.W_c_alpha, self.W_c_beta)
        else:
            outcell = cell

        if self.grad_clipping:
            outcell = theano.gradient.grad_clip(outcell, -self.grad_clipping, self.grad_clipping)

        hid = outgate*T.tanh(outcell)

        if self.num_proj_units:
            hid = T.dot(hid, self.W_p)
        
        return [cell, hid]


    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        if self.num_proj_units:
            return input_shape[0], input_shape[1], self.num_proj_units
        else:
            return input_shape[0], input_shape[1], self.num_units

    def step_masked(self, input_n, mask_n, cell_previous, hid_previous, *args):
        cell, hid = self.step(input_n, cell_previous, hid_previous, *args)

        cell = T.switch(mask_n, cell, cell_previous)
        hid = T.switch(mask_n, hid, hid_previous)

        return [cell, hid]

class LSTMLayer(LSTMOpMixin, MergeLayer):
    
    def __init__(self, incoming, num_units, 
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None,
                 num_proj_units=0,
                 use_layer_norm=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        super(LSTMLayer, self).__init__(incomings, **kwargs)

        self.num_units = num_units
     
        self.backwards = backwards
        self.grad_clipping = grad_clipping

        input_shape = self.input_shapes[0]

        self.num_inputs = numpy.prod(input_shape[2:])

        self.num_proj_units = num_proj_units
        self.use_layer_norm = use_layer_norm

        self.init_main_lstm_weights()

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
   
        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        
        seq_len, num_batch, _ = input.shape
      
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)
        self.b_stacked = T.concatenate([self.b_ig, self.b_fg, self.b_c, self.b_og], axis=0)

        input = T.dot(input, self.W_x_stacked)

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = self.step_masked
        else:
            sequences = [input]
            step_fun = self.step

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)
        
        non_seqs = [self.W_h_stacked, self.W_x_stacked, self.b_stacked]
        if self.num_proj_units:
            non_seqs.append(self.W_p)
        if self.use_layer_norm:
            non_seqs.extend([self.W_x_alpha, self.W_h_alpha, self.W_c_alpha, self.W_c_beta])

        cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[cell_init, hid_init],
            go_backwards=self.backwards,
            non_sequences=non_seqs,
            strict=True)[0]

        hid_out = hid_out.dimshuffle(1, 0, 2)

        if self.backwards:
            hid_out = hid_out[:, ::-1]

        return hid_out

class SpeakerLHUCLSTMLayer(LSTMOpMixin, MergeLayer):
    def init_lhuc_weights_helper(self, embedding_dim):
        weight_init, bias_init = init.Constant(.0), init.Constant(1.0)

        self.W_pred_list = []
        self.b_pred_list = []

        for i in range(self.num_pred_layers):
            input_dim = embedding_dim if i == 0 else self.num_pred_units

            self.W_pred_list.append(self.add_param(init.Normal(), (input_dim, self.num_pred_units),
                                name="W_e_h_{}".format(i)))
            
            self.b_pred_list.append(self.add_param(init.Constant(.0), (self.num_pred_units,), 
                                name="b_e_h_{}".format(i), regularizable=False))

        input_dim = embedding_dim if self.num_pred_layers == 0 else self.num_pred_units

        num_out_units = self.num_proj_units if self.num_proj_units else self.num_units
        self.W_pred_list.append(self.add_param(weight_init, (input_dim, num_out_units),
                            name="W_e_h_{}".format(self.num_pred_layers)))
        self.b_pred_list.append(self.add_param(bias_init, (num_out_units,),
                            name="b_e_h_{}".format(self.num_pred_layers), regularizable=False))

    def init_lhuc_weights(self):
        speaker_shape = self.input_shapes[self.speaker_incoming_index]
        speaker_dim = speaker_shape[-1]

        self.init_lhuc_weights_helper(speaker_dim)

    
    def __init__(self, incoming, speaker_input, num_units, num_pred_units, num_pred_layers, 
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
                 num_proj_units=0,
                 use_layer_norm=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if speaker_input is not None:
            incomings.append(speaker_input)
            self.speaker_incoming_index = len(incomings)-1

        super(SpeakerLHUCLSTMLayer, self).__init__(incomings, **kwargs)

        self.num_units = num_units
        self.num_pred_units = num_pred_units
        self.num_pred_layers = num_pred_layers

        self.backwards = backwards
        self.grad_clipping = grad_clipping

        input_shape = self.input_shapes[0]

        self.num_inputs = numpy.prod(input_shape[2:])

        self.reparam_fn = relu
        self.pred_act = relu

        self.num_proj_units = num_proj_units
        self.use_layer_norm = use_layer_norm

        self.init_main_lstm_weights()
        self.init_lhuc_weights()

        self.speaker_embedding = None
    
    @abstractmethod
    def compute_speaker_embedding(self, inputs):
        pass

    def compute_scaling_factor(self, speaker_embedding):
        pred_in = speaker_embedding

        for i in range(self.num_pred_layers):
            pred_in = T.dot(pred_in, self.W_pred_list[i]) + self.b_pred_list[i]
            pred_in = self.pred_act(pred_in)
            
        return T.dot(pred_in, self.W_pred_list[self.num_pred_layers]) + self.b_pred_list[self.num_pred_layers]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
   
        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        
        seq_len, num_batch, _ = input.shape

        self.speaker_embedding = self.compute_speaker_embedding(inputs)
        scaling_factor = self.compute_scaling_factor(self.speaker_embedding)
       
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)
        self.b_stacked = T.concatenate([self.b_ig, self.b_fg, self.b_c, self.b_og], axis=0)

        input = T.dot(input, self.W_x_stacked)

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = self.step_masked
        else:
            sequences = [input]
            step_fun = self.step

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)
        
        non_seqs = [self.W_h_stacked, self.W_x_stacked, self.b_stacked]
        if self.num_proj_units:
            non_seqs.append(self.W_p)
        if self.use_layer_norm:
            non_seqs.extend([self.W_x_alpha, self.W_h_alpha, self.W_c_alpha, self.W_c_beta])
        non_seqs.extend(self.W_pred_list)
        non_seqs.extend(self.b_pred_list)
      
        cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[cell_init, hid_init],
            go_backwards=self.backwards,
            non_sequences=non_seqs,
            strict=True)[0]

        # LHUC
        hid_out = hid_out * self.reparam_fn(scaling_factor)
        hid_out = hid_out.dimshuffle(1, 0, 2)

        if self.backwards:
            hid_out = hid_out[:, ::-1]

        return hid_out

    def get_speaker_embedding(self):
        return self.speaker_embedding

class SeqSumLHUCLSTMLayer(SpeakerLHUCLSTMLayer):

    def __init__(self, incoming, speaker_input, num_units, num_pred_units, num_pred_layers,
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
                 num_proj_units=0,
                 **kwargs):
       
        super(SeqSumLHUCLSTMLayer, self).__init__(incoming, 
                    speaker_input, num_units, num_pred_units, num_pred_layers, 
                    backwards, grad_clipping, mask_input, num_proj_units, **kwargs)

    def compute_speaker_embedding(self, inputs):        
        speaker_input = inputs[self.speaker_incoming_index]
        return speaker_input

class IVectorLHUCLSTMLayer(SpeakerLHUCLSTMLayer):
    def __init__(self, incoming, speaker_input, num_units, num_pred_units, num_pred_layers, 
                 backwards=False, 
                 grad_clipping=0, 
                 mask_input=None, 
                 num_proj_units=0,
                 **kwargs):

        super(IVectorLHUCLSTMLayer, self).__init__(incoming, 
                speaker_input, num_units, num_pred_units, num_pred_layers, 
                backwards, grad_clipping, mask_input, num_proj_units, **kwargs)

    def compute_speaker_embedding(self, inputs):        
        ivector_input = inputs[self.speaker_incoming_index]
        ivector_input = ivector_input.dimshuffle(1, 0, 2)
        iv_seq_len, iv_num_batch, _ = ivector_input.shape

        # we only need the ivector for the first step because they are all the same
        return ivector_input[0]


