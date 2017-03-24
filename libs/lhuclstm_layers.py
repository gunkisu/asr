import theano
import numpy
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import relu
from lasagne import init, nonlinearities
from lasagne.layers import Layer, MergeLayer, Gate

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

def ln(input, alpha, beta):
    output = (input - T.mean(input, axis=1, keepdims=True)) / T.sqrt(T.var(input, axis=1, keepdims=True) + eps)
    output = alpha[None, :] * output + beta[None, :]
    return output

from abc import ABCMeta, abstractmethod, abstractproperty

class LSTMLayer(MergeLayer):
    def slice_w(self, x, n):
        s = x[:, n*self.num_units:(n+1)*self.num_units]
        if self.num_units == 1:
            s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
        return s

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_units


    def add_gate_params(self, gate, gate_name):
        # (W_h, W_x, b)
        return (self.add_param(gate.W_hid, (self.num_units, self.num_units),
                               name="W_h_{}".format(gate_name)),
                self.add_param(gate.W_in, (self.num_inputs, self.num_units),
                               name="W_x_{}".format(gate_name)),
                self.add_param(gate.b, (self.num_units,),
                               name="b_{}".format(gate_name),
                               regularizable=False))

    def init_main_lstm_weights(self):
        (self.W_h_ig, self.W_x_ig, self.b_ig) = self.add_gate_params(self.ingate, 'ig')
        (self.W_h_fg, self.W_x_fg, self.b_fg) = self.add_gate_params(self.forgetgate, 'fg')
        (self.W_h_c, self.W_x_c, self.b_c) = self.add_gate_params(self.cell, 'c')
        (self.W_h_og, self.W_x_og, self.b_og) = self.add_gate_params(self.outgate, 'og')

    	# Peephole connections
        self.W_c_ig = self.add_param(
                self.ingate.W_cell, (self.num_units, ), name="W_c_ig")

        self.W_c_fg = self.add_param(
                self.forgetgate.W_cell, (self.num_units, ), name="W_c_fg")

        self.W_c_og = self.add_param(
                self.outgate.W_cell, (self.num_units, ), name="W_c_og")

        self.cell_init = self.add_param(self.cell_init, (1, self.num_units), name="cell_init",
            trainable=False, regularizable=False)

        self.hid_init = self.add_param(self.hid_init, (1, self.num_units), name="hid_init",
            trainable=False, regularizable=False)
    
    
    def step_masked(self, input_n, mask_n, cell_previous, hid_previous, *args):
        cell, hid = self.step(input_n, cell_previous, hid_previous, *args)

        cell = T.switch(mask_n, cell, cell_previous)
        hid = T.switch(mask_n, hid, hid_previous)

        return [cell, hid]
    
    def __init__(self, incoming, num_units, 
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        super(LSTMLayer, self).__init__(incomings, **kwargs)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
     
        self.backwards = backwards
        self.grad_clipping = grad_clipping

        input_shape = self.input_shapes[0]

        self.num_inputs = numpy.prod(input_shape[2:])

        self.ingate = ingate
        self.forgetgate = forgetgate
        self.cell = cell
        self.outgate = outgate

        self.nonlinearity_ingate = ingate.nonlinearity
        self.nonlinearity_forgetgate = forgetgate.nonlinearity
        self.nonlinearity_cell = cell.nonlinearity
        self.nonlinearity_outgate = outgate.nonlinearity

        self.cell_init = cell_init
        self.hid_init = hid_init

        self.init_main_lstm_weights()

    def step(self, input_n, cell_previous, hid_previous, *args):
        # Precomputed input outside of scan: input = T.dot(input, self.W_x_stacked) + self.b_stacked

        gates = T.dot(hid_previous, self.W_h_stacked) + input_n
   
        ingate, forgetgate, cell_input, outgate = \
            [self.slice_w(gates, i) for i in range(4)]

        # Peephole connections
        ingate += cell_previous*self.W_c_ig
        forgetgate += cell_previous*self.W_c_fg

        ingate = theano.gradient.grad_clip(ingate,
                                           -self.grad_clipping,
                                           self.grad_clipping)
        forgetgate = theano.gradient.grad_clip(forgetgate,
                                               -self.grad_clipping,
                                               self.grad_clipping)
        cell_input = theano.gradient.grad_clip(cell_input,
                                               -self.grad_clipping,
                                               self.grad_clipping)

        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        cell = forgetgate*cell_previous + ingate*cell_input

        # Peephole connection
        outgate += cell * self.W_c_og
        
        outgate = theano.gradient.grad_clip(outgate,
                                               -self.grad_clipping,
                                               self.grad_clipping)

        outgate = self.nonlinearity_outgate(outgate)

        hid = outgate*self.nonlinearity(cell)
        
        return [cell, hid]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
   
        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        
        seq_len, num_batch, _ = input.shape
      
        # Equation 12
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)
        self.b_stacked = T.concatenate([self.b_ig, self.b_fg, self.b_c, self.b_og], axis=0)

        # Precompute input
        input = T.dot(input, self.W_x_stacked) + self.b_stacked

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
        
        non_seqs = [self.W_h_stacked, self.W_x_stacked, self.b_stacked, self.W_c_ig, self.W_c_fg, self.W_c_og]
      
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

class LSTMPLayer(MergeLayer):
    def slice_w(self, x, n):
        s = x[:, n*self.num_units:(n+1)*self.num_units]
        if self.num_units == 1:
            s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
        return s

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_proj_units


    def add_gate_params(self, gate, gate_name):
        # (W_h, W_x, b)
        return (self.add_param(gate.W_hid, (self.num_proj_units, self.num_units),
                               name="W_h_{}".format(gate_name)),
                self.add_param(gate.W_in, (self.num_inputs, self.num_units),
                               name="W_x_{}".format(gate_name)),
                self.add_param(gate.b, (self.num_units,),
                               name="b_{}".format(gate_name),
                               regularizable=False))

    def init_main_lstm_weights(self):
        (self.W_h_ig, self.W_x_ig, self.b_ig) = self.add_gate_params(self.ingate, 'ig')
        (self.W_h_fg, self.W_x_fg, self.b_fg) = self.add_gate_params(self.forgetgate, 'fg')
        (self.W_h_c, self.W_x_c, self.b_c) = self.add_gate_params(self.cell, 'c')
        (self.W_h_og, self.W_x_og, self.b_og) = self.add_gate_params(self.outgate, 'og')

        self.W_p = self.add_param(Gate().W_hid, (self.num_units, self.num_proj_units), 
            name="W_p")

		# Peephole connections
        self.W_c_ig = self.add_param(
                self.ingate.W_cell, (self.num_units, ), name="W_c_ig")

        self.W_c_fg = self.add_param(
                self.forgetgate.W_cell, (self.num_units, ), name="W_c_fg")

        self.W_c_og = self.add_param(
                self.outgate.W_cell, (self.num_units, ), name="W_c_og")

        self.cell_init = self.add_param(self.cell_init, (1, self.num_units), name="cell_init",
            trainable=False, regularizable=False)

        self.hid_init = self.add_param(self.hid_init, (1, self.num_proj_units), name="hid_init",
            trainable=False, regularizable=False)
    
    
    def step_masked(self, input_n, mask_n, cell_previous, hid_previous, *args):
        cell, hid = self.step(input_n, cell_previous, hid_previous, *args)

        cell = T.switch(mask_n, cell, cell_previous)
        hid = T.switch(mask_n, hid, hid_previous)

        return [cell, hid]
    
    def __init__(self, incoming, num_units, num_proj_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        super(LSTMPLayer, self).__init__(incomings, **kwargs)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
     
        # projection layer
        self.num_proj_units = num_proj_units

        self.backwards = backwards
        self.grad_clipping = grad_clipping

        input_shape = self.input_shapes[0]

        self.num_inputs = numpy.prod(input_shape[2:])

        self.ingate = ingate
        self.forgetgate = forgetgate
        self.cell = cell
        self.outgate = outgate

        self.nonlinearity_ingate = ingate.nonlinearity
        self.nonlinearity_forgetgate = forgetgate.nonlinearity
        self.nonlinearity_cell = cell.nonlinearity
        self.nonlinearity_outgate = outgate.nonlinearity

        self.cell_init = cell_init
        self.hid_init = hid_init

        self.init_main_lstm_weights()

    def step(self, input_n, cell_previous, hid_previous, *args):
        # Precomputed input outside of scan: input = T.dot(input, self.W_x_stacked) + self.b_stacked

        gates = T.dot(hid_previous, self.W_h_stacked) + input_n
 
        ingate, forgetgate, cell_input, outgate = \
            [self.slice_w(gates, i) for i in range(4)]

        # Peephole connections
        ingate += cell_previous*self.W_c_ig
        forgetgate += cell_previous*self.W_c_fg

        ingate = theano.gradient.grad_clip(ingate,
                                           -self.grad_clipping,
                                           self.grad_clipping)
        forgetgate = theano.gradient.grad_clip(forgetgate,
                                               -self.grad_clipping,
                                               self.grad_clipping)
        cell_input = theano.gradient.grad_clip(cell_input,
                                               -self.grad_clipping,
                                               self.grad_clipping)

        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        cell = forgetgate*cell_previous + ingate*cell_input

        # Peephole connections
        outgate += cell * self.W_c_og
        outgate = theano.gradient.grad_clip(outgate,
                                               -self.grad_clipping,
                                               self.grad_clipping)

        outgate = self.nonlinearity_outgate(outgate)
        hid = outgate*self.nonlinearity(cell)
        
        # projection layer
        hid = T.dot(hid, self.W_p) 
       
        return [cell, hid]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
   
        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        
        seq_len, num_batch, _ = input.shape
      
        # Equation 12
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)
        self.b_stacked = T.concatenate([self.b_ig, self.b_fg, self.b_c, self.b_og], axis=0)

        # Precompute input
        input = T.dot(input, self.W_x_stacked) + self.b_stacked

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
        
        non_seqs = [self.W_h_stacked, self.W_x_stacked, self.b_stacked, self.W_c_ig, self.W_c_fg, self.W_c_og, self.W_p]
      
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

class SpeakerLHUCLSTMLayer(MergeLayer):
    __metaclass__ = ABCMeta

    def slice_w(self, x, n):
        s = x[:, n*self.num_units:(n+1)*self.num_units]
        if self.num_units == 1:
            s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
        return s

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_units


    def add_gate_params(self, gate, gate_name):
        # (W_h, W_x, b)
        return (self.add_param(gate.W_hid, (self.num_units, self.num_units),
                               name="W_h_{}".format(gate_name)),
                self.add_param(gate.W_in, (self.num_inputs, self.num_units),
                               name="W_x_{}".format(gate_name)),
                self.add_param(gate.b, (self.num_units,),
                               name="b_{}".format(gate_name),
                               regularizable=False))

    def init_main_lstm_weights(self):
        (self.W_h_ig, self.W_x_ig, self.b_ig) = self.add_gate_params(self.ingate, 'ig')
        (self.W_h_fg, self.W_x_fg, self.b_fg) = self.add_gate_params(self.forgetgate, 'fg')
        (self.W_h_c, self.W_x_c, self.b_c) = self.add_gate_params(self.cell, 'c')
        (self.W_h_og, self.W_x_og, self.b_og) = self.add_gate_params(self.outgate, 'og')

		# Peephole connections
        self.W_c_ig = self.add_param(
                self.ingate.W_cell, (self.num_units, ), name="W_c_ig")

        self.W_c_fg = self.add_param(
                self.forgetgate.W_cell, (self.num_units, ), name="W_c_fg")

        self.W_c_og = self.add_param(
                self.outgate.W_cell, (self.num_units, ), name="W_c_og")

        self.cell_init = self.add_param(self.cell_init, (1, self.num_units), name="cell_init",
            trainable=False, regularizable=False)

        self.hid_init = self.add_param(self.hid_init, (1, self.num_units), name="hid_init",
            trainable=False, regularizable=False)
    
    
    def step_masked(self, input_n, mask_n, cell_previous, hid_previous, *args):
        cell, hid = self.step(input_n, cell_previous, hid_previous, *args)

        cell = T.switch(mask_n, cell, cell_previous)
        hid = T.switch(mask_n, hid, hid_previous)

        return [cell, hid]

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
        self.W_pred_list.append(self.add_param(weight_init, (input_dim, self.num_units),
                            name="W_e_h_{}".format(self.num_pred_layers)))
        self.b_pred_list.append(self.add_param(bias_init, (self.num_units,),
                            name="b_e_h_{}".format(self.num_pred_layers), regularizable=False))

    def init_lhuc_weights(self):
        speaker_shape = self.input_shapes[self.speaker_incoming_index]
        speaker_dim = speaker_shape[-1]

        self.init_lhuc_weights_helper(speaker_dim)

    
    def __init__(self, incoming, speaker_input, num_units, num_pred_units, num_pred_layers, 
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
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

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.num_pred_units = num_pred_units
        self.num_pred_layers = num_pred_layers

        self.backwards = backwards
        self.grad_clipping = grad_clipping

        input_shape = self.input_shapes[0]

        self.num_inputs = numpy.prod(input_shape[2:])

        self.ingate = ingate
        self.forgetgate = forgetgate
        self.cell = cell
        self.outgate = outgate

        self.nonlinearity_ingate = ingate.nonlinearity
        self.nonlinearity_forgetgate = forgetgate.nonlinearity
        self.nonlinearity_cell = cell.nonlinearity
        self.nonlinearity_outgate = outgate.nonlinearity

        self.cell_init = cell_init
        self.hid_init = hid_init

        self.reparam_fn = relu
        self.pred_act = relu

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

    def step(self, input_n, cell_previous, hid_previous, *args):
        # Precomputed input outside of scan: input = T.dot(input, self.W_x_stacked) + self.b_stacked

        gates = T.dot(hid_previous, self.W_h_stacked) + input_n
   
        ingate, forgetgate, cell_input, outgate = \
            [self.slice_w(gates, i) for i in range(4)]

        # Peephole connections
        ingate += cell_previous*self.W_c_ig
        forgetgate += cell_previous*self.W_c_fg

        ingate = theano.gradient.grad_clip(ingate,
                                           -self.grad_clipping,
                                           self.grad_clipping)
        forgetgate = theano.gradient.grad_clip(forgetgate,
                                               -self.grad_clipping,
                                               self.grad_clipping)
        cell_input = theano.gradient.grad_clip(cell_input,
                                               -self.grad_clipping,
                                               self.grad_clipping)

        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        cell = forgetgate*cell_previous + ingate*cell_input

        # Peephole connections
        outgate += cell * self.W_c_og
        outgate = theano.gradient.grad_clip(outgate,
                                               -self.grad_clipping,
                                               self.grad_clipping)

        outgate = self.nonlinearity_outgate(outgate)

        hid = outgate*self.nonlinearity(cell)
     
        return [cell, hid]

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
       
        # Equation 12
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)
        self.b_stacked = T.concatenate([self.b_ig, self.b_fg, self.b_c, self.b_og], axis=0)

        # Precompute input
        input = T.dot(input, self.W_x_stacked) + self.b_stacked

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
        
        non_seqs = [self.W_h_stacked, self.W_x_stacked, self.b_stacked, self.W_c_ig, self.W_c_fg, self.W_c_og]
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
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
                 **kwargs):
       
        super(SeqSumLHUCLSTMLayer, self).__init__(incoming, speaker_input, num_units, num_pred_units, num_pred_layers, 
                 ingate, forgetgate, cell, outgate, nonlinearity, cell_init, hid_init, backwards,
                 grad_clipping, mask_input, **kwargs)

    def compute_speaker_embedding(self, inputs):        
        speaker_input = inputs[self.speaker_incoming_index]
        return speaker_input

class IVectorLHUCLSTMLayer(SpeakerLHUCLSTMLayer):
    def __init__(self, incoming, speaker_input, num_units, num_pred_units, num_pred_layers, 
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, reparam='relu', 
                 **kwargs):

        super(IVectorLHUCLSTMLayer, self).__init__(incoming, speaker_input, num_units, num_pred_units, num_pred_layers, 
                 ingate, forgetgate, cell, outgate, nonlinearity, cell_init, hid_init, backwards,
                 grad_clipping, mask_input, **kwargs)

    def compute_speaker_embedding(self, inputs):        
        ivector_input = inputs[self.speaker_incoming_index]
        ivector_input = ivector_input.dimshuffle(1, 0, 2)
        iv_seq_len, iv_num_batch, _ = ivector_input.shape

        # we only need the ivector for the first step because they are all the same
        return ivector_input[0]

