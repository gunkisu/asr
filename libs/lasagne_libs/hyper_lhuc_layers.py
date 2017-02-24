import theano
import numpy
from theano import tensor as T
from lasagne import init, nonlinearities
from lasagne.layers import Layer, MergeLayer, Gate
floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

def ln(input, alpha, beta):
    output = (input - T.mean(input, axis=1, keepdims=True)) / T.sqrt(T.var(input, axis=1, keepdims=True) + eps)
    output = alpha[None, :] * output + beta[None, :]
    return output

def reparam_2sigmoid(scaling_factor):
    return 2/(1+T.exp(-scaling_factor))

def reparam_exp(scaling_factor):
    return T.exp(scaling_factor)

def reparam_identity(scaling_factor):
    return scaling_factor

def reparam_relu(scaling_factor):
    return T.nnet.relu(scaling_factor)

def to_reparam_fn(fn_name):
    return eval('reparam_{}'.format(fn_name))

def weight_init(fn_name):
    if fn_name == '2sigmoid':
        return init.Constant(.0), init.Constant(.0) 
    else:
        return init.Constant(.0), init.Constant(1.0)


class HyperLSTMLayer(MergeLayer):
    def add_scale_params(self, gate_name):
        # (W_hz, W_xz, W_bz, b_0)
        return (self.add_param(init.Constant(1.0/self.num_proj_units), (self.num_proj_units, self.num_units),
                               name="W_hz_{}".format(gate_name)),
                self.add_param(init.Constant(1.0/self.num_proj_units), (self.num_proj_units, self.num_units),
                               name="W_xz_{}".format(gate_name)),
                self.add_param(init.Constant(0.), (self.num_proj_units, self.num_units),
                               name="W_bz_{}".format(gate_name)),
                self.add_param(init.Constant(0.), (self.num_units,),
                               name="b_0_{}".format(gate_name),
                               regularizable=False))

    def add_hyper_gate_params(self, gate, gate_name):
        # (W_hhat, W_xhat, bhat)
        return (self.add_param(gate.W_hid, (self.num_hyper_units, self.num_hyper_units),
                               name="W_hhat_{}".format(gate_name)),
                self.add_param(gate.W_in, (self.num_inputs+self.num_units, self.num_hyper_units),
                               name="W_xhat_{}".format(gate_name)),
                self.add_param(gate.b, (self.num_hyper_units,),
                               name="bhat_{}".format(gate_name),
                               regularizable=False))

    def add_proj_params(self, gate_name):
        """Initalization as described in the paper"""

        # (W_hhat_h, b_hhat, W_hhat_x, b_hhat_x, W_hhat_b)
        return (self.add_param(init.Constant(0.), (self.num_hyper_units, self.num_proj_units),
                               name="W_hhat_h_{}".format(gate_name)),
                self.add_param(init.Constant(1.), (self.num_proj_units,),
                               name="b_hhat_h_{}".format(gate_name),
                               regularizable=False),
                self.add_param(init.Constant(0.), (self.num_hyper_units, self.num_proj_units),
                               name="W_hhat_x_{}".format(gate_name)),
                self.add_param(init.Constant(1.), (self.num_proj_units,),
                               name="b_hhat_x_{}".format(gate_name),
                               regularizable=False),
                self.add_param(init.Constant(0.), (self.num_hyper_units, self.num_proj_units),
                               name="W_hhat_b_{}".format(gate_name)))

    def add_gate_params(self, gate, gate_name):
        # (W_h, W_x)
        return (self.add_param(gate.W_hid, (self.num_units, self.num_units),
                               name="W_h_{}".format(gate_name)),
                self.add_param(gate.W_in, (self.num_inputs, self.num_units),
                               name="W_x_{}".format(gate_name))
               )

    def compute_eq10(self, hid_previous, orig_input_n, W_xhat_stacked, 
            bhat_stacked, hyper_cell_previous, hyper_hid_previous, W_hhat_stacked):
        hyper_input_n = T.concatenate([hid_previous, orig_input_n], axis=1)

        hyper_input_n = T.dot(hyper_input_n, W_xhat_stacked) + bhat_stacked
        hyper_gates = hyper_input_n + T.dot(hyper_hid_previous, W_hhat_stacked)
        if self.grad_clipping:
            hyper_gates = theano.gradient.grad_clip(
                hyper_gates, -self.grad_clipping, self.grad_clipping)
        
        hyper_ig, hyper_fg, hyper_cell_input, hyper_og = \
            [self.hyper_slice(hyper_gates, i) for i in range(4)]
        hyper_ig = self.nonlinearity_ingate(hyper_ig)
        hyper_fg = self.nonlinearity_forgetgate(hyper_fg)
        hyper_cell_input = self.nonlinearity_cell(hyper_cell_input)

        hyper_cell = hyper_fg * hyper_cell_previous + hyper_ig * hyper_cell_input
        hyper_og = self.nonlinearity_outgate(hyper_og)

        hyper_hid = hyper_og * self.nonlinearity(hyper_cell)

        return [hyper_cell, hyper_hid]        

    def init_main_lstm_weights(self):
        (self.W_h_ig, self.W_x_ig) = self.add_gate_params(self.ingate, 'ig')
        (self.W_h_fg, self.W_x_fg) = self.add_gate_params(self.forgetgate, 'fg')
        (self.W_h_c, self.W_x_c) = self.add_gate_params(self.cell, 'c')
        (self.W_h_og, self.W_x_og) = self.add_gate_params(self.outgate, 'og')

        self.cell_init = self.add_param(self.cell_init, (1, self.num_units), name="cell_init",
            trainable=False, regularizable=False)

        self.hid_init = self.add_param(self.hid_init, (1, self.num_units), name="hid_init",
            trainable=False, regularizable=False)

    def init_weights(self):
        # Equation 10
        (self.W_hhat_ig, self.W_xhat_ig, self.bhat_ig) = self.add_hyper_gate_params(self.ingate, 'ig')
        (self.W_hhat_fg, self.W_xhat_fg, self.bhat_fg) = self.add_hyper_gate_params(self.forgetgate, 'fg')
        (self.W_hhat_c, self.W_xhat_c, self.bhat_c) = self.add_hyper_gate_params(self.cell, 'c')
        (self.W_hhat_og, self.W_xhat_og, self.bhat_og) = self.add_hyper_gate_params(self.outgate, 'og')

        self.hyper_cell_init = self.add_param(self.cell_init, (1, self.num_hyper_units), name="hyper_cell_init",
            trainable=False, regularizable=False)

        self.hyper_hid_init = self.add_param(self.hid_init, (1, self.num_hyper_units), name="hyper_hid_init",
            trainable=False, regularizable=False)

        # Equation 11
        (self.W_hhat_h_ig, self.b_hhat_h_ig, self.W_hhat_x_ig, self.b_hhat_x_ig, self.W_hhat_b_ig) = self.add_proj_params('ig')
        (self.W_hhat_h_fg, self.b_hhat_h_fg, self.W_hhat_x_fg, self.b_hhat_x_fg, self.W_hhat_b_fg) = self.add_proj_params('fg')
        (self.W_hhat_h_c, self.b_hhat_h_c, self.W_hhat_x_c, self.b_hhat_x_c, self.W_hhat_b_c) = self.add_proj_params('c')
        (self.W_hhat_h_og, self.b_hhat_h_og, self.W_hhat_x_og, self.b_hhat_x_og, self.W_hhat_b_og) = self.add_proj_params('og')

        # Equation 12
        (self.W_hz_ig, self.W_xz_ig, self.W_bz_ig, self.b_0_ig) = self.add_scale_params('ig')
        (self.W_hz_fg, self.W_xz_fg, self.W_bz_fg, self.b_0_fg) = self.add_scale_params('fg')
        (self.W_hz_c, self.W_xz_c, self.W_bz_c, self.b_0_c) = self.add_scale_params('c')
        (self.W_hz_og, self.W_xz_og, self.W_bz_og, self.b_0_og) = self.add_scale_params('og')

        self.init_main_lstm_weights()

    def slice_w(self, x, n):
        s = x[:, n*self.num_units:(n+1)*self.num_units]
        if self.num_units == 1:
            s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
        return s

    def hyper_slice(self, x, n):
        s = x[:, n*self.num_hyper_units:(n+1)*self.num_hyper_units]
        if self.num_hyper_units == 1:
            s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
        return s

    def proj_slice(self, x, n):
        s = x[:, n*self.num_proj_units:(n+1)*self.num_proj_units]
        if self.num_proj_units == 1:
            s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
        return s

    def __init__(self, incoming, num_units, num_hyper_units,num_proj_units,
                 ingate=Gate(W_in=init.Orthogonal()),
                 forgetgate=Gate(W_in=init.Orthogonal()),
                 cell=Gate(W_in=init.Orthogonal(), W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(W_in=init.Orthogonal()),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 precompute_input=True,
                 mask_input=None, 
                 ivector_input=None, use_layer_norm=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if ivector_input is not None:
            incomings.append(ivector_input)
            self.ivector_incoming_index = len(incomings)-1

        super(HyperLSTMLayer, self).__init__(incomings, **kwargs)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.num_hyper_units = num_hyper_units
        self.num_proj_units = num_proj_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.precompute_input = precompute_input

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

        self.use_layer_norm = use_layer_norm

        self.init_weights()

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_units

    def step(self, orig_input_n, input_n, hyper_cell_previous, hyper_hid_previous, cell_previous, hid_previous, *args):
        hyper_cell, hyper_hid = self.compute_eq10(hid_previous, orig_input_n, self.W_xhat_stacked,
            self.bhat_stacked, hyper_cell_previous, hyper_hid_previous, self.W_hhat_stacked)
     
        # Equation 11
        z_h = T.dot(hyper_hid, self.W_hhat_h_stacked) + self.b_hhat_h_stacked
        z_x = T.dot(hyper_hid, self.W_hhat_x_stacked) + self.b_hhat_x_stacked
        z_b = T.dot(hyper_hid, self.W_hhat_b_stacked)

        # Equation 12
        z_h_list = [self.proj_slice(z_h, i) for i in range(4)]
        z_x_list = [self.proj_slice(z_x, i) for i in range(4)]
        z_b_list = [self.proj_slice(z_b, i) for i in range(4)]

        d_h_list = []
        d_x_list = []
        b_list = []
        for z_h_slice, z_x_slice, z_b_slice, W_hz_slice, W_xz_slice, W_bz_slice, b_0_slice in \
                zip(z_h_list, z_x_list, z_b_list, self.W_hz_list, self.W_xz_list, self.W_bz_list, self.b_0_list):
            d_h_list.append(T.dot(z_h_slice, W_hz_slice))
            d_x_list.append(T.dot(z_x_slice, W_xz_slice))
            b_list.append(T.dot(z_b_slice, W_bz_slice)+b_0_slice)
        
        d_h_stacked = T.concatenate(d_h_list, axis=1)
        d_x_stacked = T.concatenate(d_x_list, axis=1)
        b_stacked = T.concatenate(b_list, axis=1)
        
        if not self.precompute_input:
            input_n = T.dot(input_n, self.W_x_stacked)
        
        input_n = input_n * d_x_stacked + b_stacked
        gates = T.dot(hid_previous, self.W_h_stacked)
        gates = gates * d_h_stacked + input_n
   
        if self.grad_clipping:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        ingate, forgetgate, cell_input, outgate = \
            [self.slice_w(gates, i) for i in range(4)]
    
        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        cell = forgetgate*cell_previous + ingate*cell_input
        outgate = self.nonlinearity_outgate(outgate)

        hid = outgate*self.nonlinearity(cell)
        return [hyper_cell, hyper_hid, cell, hid]

    def step_masked(self, orig_input_n, input_n, mask_n, hyper_cell_previous, 
                hyper_hid_previous, cell_previous, hid_previous, *args):
        hyper_cell, hyper_hid, cell, hid = self.step(orig_input_n, input_n, 
            hyper_cell_previous, hyper_hid_previous, cell_previous, hid_previous, *args)

        hyper_cell = T.switch(mask_n, hyper_cell, hyper_cell_previous)
        hyper_hid = T.switch(mask_n, hyper_hid, hyper_hid_previous)

        cell = T.switch(mask_n, cell, cell_previous)
        hid = T.switch(mask_n, hid, hid_previous)

        return [hyper_cell, hyper_hid, cell, hid]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        hyper_hid_init = None
        hyper_cell_init = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
   
        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)

        orig_input = input 

        seq_len, num_batch, _ = input.shape

        # Equation 12
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)

        self.W_hz_list = [self.W_hz_ig, self.W_hz_fg, self.W_hz_c, self.W_hz_og]
        self.W_xz_list = [self.W_xz_ig, self.W_xz_fg, self.W_xz_c, self.W_xz_og]

        self.W_bz_list = [self.W_bz_ig, self.W_bz_fg, self.W_bz_c, self.W_bz_og]
        self.b_0_list = [self.b_0_ig, self.b_0_fg, self.b_0_c, self.b_0_og]

        # Equation 10
        self.W_hhat_stacked = T.concatenate([self.W_hhat_ig, self.W_hhat_fg, self.W_hhat_c, self.W_hhat_og], axis=1)
        self.W_xhat_stacked = T.concatenate([self.W_xhat_ig, self.W_xhat_fg, self.W_xhat_c, self.W_xhat_og], axis=1)
        self.bhat_stacked = T.concatenate([self.bhat_ig, self.bhat_fg, self.bhat_c, self.bhat_og], axis=0)

        # Equation 11
        self.W_hhat_h_stacked = T.concatenate([self.W_hhat_h_ig, self.W_hhat_h_fg, self.W_hhat_h_c, self.W_hhat_h_og], axis=1)
        self.W_hhat_x_stacked = T.concatenate([self.W_hhat_x_ig, self.W_hhat_x_fg, self.W_hhat_x_c, self.W_hhat_x_og], axis=1)
        self.W_hhat_b_stacked = T.concatenate([self.W_hhat_b_ig, self.W_hhat_b_fg, self.W_hhat_b_c, self.W_hhat_b_og], axis=1)
        self.b_hhat_h_stacked = T.concatenate([self.b_hhat_h_ig, self.b_hhat_h_fg, self.b_hhat_h_c, self.b_hhat_h_og], axis=0)
        self.b_hhat_x_stacked = T.concatenate([self.b_hhat_x_ig, self.b_hhat_x_fg, self.b_hhat_x_c, self.b_hhat_x_og], axis=0)

        if self.precompute_input:
            input = T.dot(input, self.W_x_stacked)

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [orig_input, input, mask]
            step_fun = self.step_masked
        else:
            sequences = [orig_input, input]
            step_fun = self.step

        ones = T.ones((num_batch, 1))
        hyper_cell_init = T.dot(ones, self.hyper_cell_init)
        hyper_hid_init = T.dot(ones, self.hyper_hid_init)
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [self.W_xhat_stacked, self.bhat_stacked, self.W_hhat_stacked, self.W_hhat_h_stacked, 
            self.b_hhat_h_stacked, self.W_hhat_x_stacked, self.b_hhat_x_stacked, self.W_hhat_b_stacked,
            self.W_h_stacked, self.W_x_stacked]
        non_seqs.extend(self.W_hz_list)
        non_seqs.extend(self.W_xz_list)
        non_seqs.extend(self.W_bz_list)
        non_seqs.extend(self.b_0_list)

        hyper_cell_out, hyper_hid_out, cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[hyper_cell_init, hyper_hid_init, cell_init, hid_init],
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]

        hid_out = hid_out.dimshuffle(1, 0, 2)

        if self.backwards:
            hid_out = hid_out[:, ::-1]

        return hid_out

class HyperLHUCLSTMLayer(HyperLSTMLayer):

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

        self.cell_init = self.add_param(self.cell_init, (1, self.num_units), name="cell_init",
            trainable=False, regularizable=False)

        self.hid_init = self.add_param(self.hid_init, (1, self.num_units), name="hid_init",
            trainable=False, regularizable=False)

    def init_weights(self):
        (self.W_hhat_ig, self.W_xhat_ig, self.bhat_ig) = self.add_hyper_gate_params(self.ingate, 'ig')
        (self.W_hhat_fg, self.W_xhat_fg, self.bhat_fg) = self.add_hyper_gate_params(self.forgetgate, 'fg')
        (self.W_hhat_c, self.W_xhat_c, self.bhat_c) = self.add_hyper_gate_params(self.cell, 'c')
        (self.W_hhat_og, self.W_xhat_og, self.bhat_og) = self.add_hyper_gate_params(self.outgate, 'og')

        self.hyper_cell_init = self.add_param(self.cell_init, (1, self.num_hyper_units), name="hyper_cell_init",
            trainable=False, regularizable=False)

        self.hyper_hid_init = self.add_param(self.hid_init, (1, self.num_hyper_units), name="hyper_hid_init",
            trainable=False, regularizable=False)

        # Equation 11
        self.W_hhat_h = self.add_param(init.Constant(0.), (self.num_hyper_units, self.num_proj_units),
                        name="W_hhat_h")
        self.b_hhat_h = self.add_param(init.Constant(1.), (self.num_proj_units,),
                                   name="b_hhat_h", regularizable=False)
   
        # Equation 12
        self.W_hz = self.add_param(init.Constant(1.0/self.num_proj_units), (self.num_proj_units, self.num_units),
                                   name="W_hz")

        self.init_main_lstm_weights()
    
    def __init__(self, incoming, num_units, num_hyper_units,num_proj_units,
                 ingate=Gate(W_in=init.Orthogonal()),
                 forgetgate=Gate(W_in=init.Orthogonal()),
                 cell=Gate(W_in=init.Orthogonal(), W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(W_in=init.Orthogonal()),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 precompute_input=True,
                 mask_input=None,
                 reparam='relu', use_layer_norm=False,
                 **kwargs):

        super(HyperLHUCLSTMLayer, self).__init__(incoming, num_units, num_hyper_units,num_proj_units,
                 ingate, forgetgate, cell, outgate, nonlinearity, cell_init, hid_init, backwards,
                 gradient_steps, grad_clipping, precompute_input, mask_input, use_layer_norm=use_layer_norm, **kwargs)


        self.reparam = to_reparam_fn(reparam)
        
    def step(self, orig_input_n, input_n, hyper_cell_previous, hyper_hid_previous, cell_previous, hid_previous, *args):
        hyper_cell, hyper_hid = self.compute_eq10(hid_previous, orig_input_n, self.W_xhat_stacked,
            self.bhat_stacked, hyper_cell_previous, hyper_hid_previous, self.W_hhat_stacked)
        
        # Equation 11
        z_h = T.dot(hyper_hid, self.W_hhat_h) + self.b_hhat_h

        # Equation 12
        d_h = T.dot(z_h, self.W_hz)

        if not self.precompute_input:
            input_n = T.dot(input_n, self.W_x_stacked) + self.b_stacked
        
        gates = T.dot(hid_previous, self.W_h_stacked) + input_n
   
        if self.grad_clipping:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        ingate, forgetgate, cell_input, outgate = \
            [self.slice_w(gates, i) for i in range(4)]
    
        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        cell = forgetgate*cell_previous + ingate*cell_input
        outgate = self.nonlinearity_outgate(outgate)

        hid = outgate*self.nonlinearity(cell)
        
        # LHUC
        hid = hid * self.reparam(d_h)
        return [hyper_cell, hyper_hid, cell, hid]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        hyper_hid_init = None
        hyper_cell_init = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
   
        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        
        orig_input = input 

        seq_len, num_batch, _ = input.shape
        
        # Equation 10
        self.W_hhat_stacked = T.concatenate([self.W_hhat_ig, self.W_hhat_fg, self.W_hhat_c, self.W_hhat_og], axis=1)
        self.W_xhat_stacked = T.concatenate([self.W_xhat_ig, self.W_xhat_fg, self.W_xhat_c, self.W_xhat_og], axis=1)
        self.bhat_stacked = T.concatenate([self.bhat_ig, self.bhat_fg, self.bhat_c, self.bhat_og], axis=0)

        # Equation 12
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)
        self.b_stacked = T.concatenate([self.b_ig, self.b_fg, self.b_c, self.b_og], axis=0)

        if self.precompute_input:
            input = T.dot(input, self.W_x_stacked) + self.b_stacked

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [orig_input, input, mask]
            step_fun = self.step_masked
        else:
            sequences = [orig_input, input]
            step_fun = self.step

        ones = T.ones((num_batch, 1))
        hyper_cell_init = T.dot(ones, self.hyper_cell_init)
        hyper_hid_init = T.dot(ones, self.hyper_hid_init)
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [self.W_xhat_stacked, self.W_hhat_stacked, self.bhat_stacked, self.W_hhat_h, 
            self.b_hhat_h, self.W_h_stacked, self.W_x_stacked, self.b_stacked, self.W_hz]

        hyper_cell_out, hyper_hid_out, cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[hyper_cell_init, hyper_hid_init, cell_init, hid_init],
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]

        hid_out = hid_out.dimshuffle(1, 0, 2)

        if self.backwards:
            hid_out = hid_out[:, ::-1]

        return hid_out

#
# Non-hyper Layers
#
from abc import ABCMeta, abstractmethod, abstractproperty

class BaseLHUCLSTMLayer(MergeLayer):
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

        self.cell_init = self.add_param(self.cell_init, (1, self.num_units), name="cell_init",
            trainable=False, regularizable=False)

        self.hid_init = self.add_param(self.hid_init, (1, self.num_units), name="hid_init",
            trainable=False, regularizable=False)
    
    
    def step_masked(self, input_n, mask_n, cell_previous, hid_previous, *args):
        cell, hid = self.step(input_n, cell_previous, hid_previous, *args)

        cell = T.switch(mask_n, cell, cell_previous)
        hid = T.switch(mask_n, hid, hid_previous)

        return [cell, hid]

    @abstractmethod
    def init_lhuc_weights(self):
        pass

    def __init__(self, incoming, num_units,
                 ingate=Gate(W_in=init.Orthogonal()),
                 forgetgate=Gate(W_in=init.Orthogonal()),
                 cell=Gate(W_in=init.Orthogonal(), W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(W_in=init.Orthogonal()),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
                 ivector_input=None,
                 reparam='relu', 
                 use_layer_norm=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if ivector_input is not None:
            incomings.append(ivector_input)
            self.ivector_incoming_index = len(incomings)-1

        super(BaseLHUCLSTMLayer, self).__init__(incomings, **kwargs)

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

        self.reparam_fn = to_reparam_fn(reparam)
        self.reparam_weight_init = weight_init(reparam)

        self.use_layer_norm = use_layer_norm


        self.init_main_lstm_weights()
        self.init_lhuc_weights()
    
    @abstractmethod
    def compute_speaker_embedding(self, inputs):
        pass

    @abstractmethod
    def compute_scaling_factor(self, speaker_embedding):
        pass

    def step(self, input_n, cell_previous, hid_previous, *args):
        # Precomputed input outside of scan: input = T.dot(input, self.W_x_stacked) + self.b_stacked

        gates = T.dot(hid_previous, self.W_h_stacked) + input_n
   
        if self.grad_clipping:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        ingate, forgetgate, cell_input, outgate = \
            [self.slice_w(gates, i) for i in range(4)]
    
        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        cell = forgetgate*cell_previous + ingate*cell_input
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

        speaker_embedding = self.compute_speaker_embedding(inputs)
        scaling_factor = self.compute_scaling_factor(speaker_embedding)
       
        # Equation 12
        self.W_h_stacked = T.concatenate([self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        self.W_x_stacked = T.concatenate([self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)
        self.b_stacked = T.concatenate([self.b_ig, self.b_fg, self.b_c, self.b_og], axis=0)

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
        
        non_seqs = [self.W_h_stacked, self.W_x_stacked, self.b_stacked]
      
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

class SummarizingLHUCLSTMLayer(BaseLHUCLSTMLayer):

    def init_lhuc_weights(self):
        weight_init, bias_init = self.reparam_weight_init

        self.W_e_h = self.add_param(weight_init, (self.num_inputs, self.num_units),
                            name="W_e_h")
        self.b_e_h = self.add_param(bias_init, (self.num_units,),
                            name="b_e_h", regularizable=False)



    def __init__(self, incoming, num_units,
                 ingate=Gate(W_in=init.Orthogonal()),
                 forgetgate=Gate(W_in=init.Orthogonal()),
                 cell=Gate(W_in=init.Orthogonal(), W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(W_in=init.Orthogonal()),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 grad_clipping=0,
                 mask_input=None, 
                 reparam='relu', 
                 use_layer_norm=False,
                 **kwargs):

        super(SummarizingLHUCLSTMLayer, self).__init__(incoming, num_units,
                 ingate, forgetgate, cell, outgate, nonlinearity, cell_init, hid_init, backwards,
                 grad_clipping, mask_input, reparam=reparam, use_layer_norm=use_layer_norm, **kwargs)

    def compute_speaker_embedding(self, inputs):
        input = inputs[0]

        input = input.dimshuffle(1, 0, 2)
        
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        if mask is not None:
            # n_seq, n_batch
            mask_for_mean = mask.dimshuffle(1,0)
            # n_batch
            seq_len = T.sum(mask_for_mean, axis=0)
            # n_batch, 1
            seq_len = seq_len[:,None]
            # n_batch, n_dim
            seq_sum = T.sum(input, axis=0)
            return seq_sum / seq_len
        else:
            return T.mean(input, axis=0)
            
    def compute_scaling_factor(self, speaker_embedding):

        return T.dot(speaker_embedding, self.W_e_h) + self.b_e_h


class IVectorLHUCLSTMLayer(SummarizingLHUCLSTMLayer):

    ''' Generates scaling vectors based on ivectors'''
 
    def __init__(self, incoming, ivector_input, num_units,
                 ingate=Gate(W_in=init.Orthogonal()),
                 forgetgate=Gate(W_in=init.Orthogonal()),
                 cell=Gate(W_in=init.Orthogonal(), W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(W_in=init.Orthogonal()),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 
                 grad_clipping=0,
             
                 mask_input=None, reparam='relu', use_layer_norm=False,
                 **kwargs):

        super(IVectorLHUCLSTMLayer, self).__init__(incoming, num_units,
                 ingate, forgetgate, cell, outgate, nonlinearity, cell_init, hid_init, backwards,
                 grad_clipping, mask_input, 
                 ivector_input=ivector_input, reparam=reparam, use_layer_norm=use_layer_norm, **kwargs)


    def init_lhuc_weights(self):
        ivector_shape = self.input_shapes[self.ivector_incoming_index]
        self.ivector_dim = ivector_shape[-1]

        weight_init, bias_init = self.reparam_weight_init

        self.W_e_h = self.add_param(weight_init, (self.ivector_dim, self.num_units),
                            name="W_e_h")
        self.b_e_h = self.add_param(bias_init, (self.num_units,),
                            name="b_e_h", regularizable=False)


    def compute_speaker_embedding(self, inputs):        
        ivector_input = inputs[self.ivector_incoming_index]
        ivector_input = ivector_input.dimshuffle(1, 0, 2)
        iv_seq_len, iv_num_batch, _ = ivector_input.shape

        # we only need the ivector for the first step because they are all the same
        return ivector_input[0]
       
