import numpy
import theano
from theano import tensor as T
from lasagne.layers import MergeLayer
from lasagne import init

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

def ln(input, alpha, beta=None):
    output = (input - T.mean(input, axis=1, keepdims=True)) / T.sqrt(T.var(input, axis=1, keepdims=True) + eps)
    output = alpha[None, :]* output
    if beta:
        output += beta[None, :]
    return output

def gate_ln(input, alpha, beta=None):
    gate_alpha = T.reshape(alpha, (1, 4, -1))
    gate_input = T.reshape(input, (input.shape[0], 4, -1))
    gate_output = (gate_input - T.mean(gate_input, axis=2, keepdims=True)) / T.sqrt(T.var(gate_input, axis=2, keepdims=True) + eps)
    gate_output = gate_alpha*gate_output
    if beta:
        gate_beta = T.reshape(beta, (1, 4, -1))
        gate_output += gate_beta
    return T.reshape(gate_output, (gate_output.shape[0], -1))

def sample_ln(input, alpha, beta=None):
    output = (input - T.mean(input, axis=1, keepdims=True)) / T.sqrt(T.var(input, axis=1, keepdims=True) + eps)
    output = alpha*output
    if beta:
        output += beta
    return output

def sample_gate_ln(input, alpha, beta=None):
    gate_alpha = T.reshape(alpha, (alpha.shape[0], 4, -1))
    gate_input = T.reshape(input, (input.shape[0], 4, -1))
    gate_output = (gate_input - T.mean(gate_input, axis=2, keepdims=True)) / T.sqrt(T.var(gate_input, axis=2, keepdims=True) + eps)
    gate_output = gate_alpha * gate_output
    if beta:
        gate_beta = T.reshape(beta, (beta.shape[0], 4, -1))
        gate_output += gate_beta
    return T.reshape(gate_output, (gate_output.shape[0], -1))


class LSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_factors=None,
                 backwards=False,
                 learn_init=False,
                 peepholes=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        hid_init = init.Constant(0.)
        cell_init = init.Constant(0.)

        super(LSTMLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        input_shape = self.input_shapes[0]
        num_inputs = input_shape[2]

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(spec=init.Uniform(0.1),
                                                   shape=(num_units, ),
                                                   name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(spec=init.Uniform(0.1),
                                                       shape=(num_units, ),
                                                       name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(spec=init.Uniform(0.1),
                                                    shape=(num_units, ),
                                                    name="W_cell_to_outgate")

        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=learn_init,
                                        regularizable=False)

        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units),
                                       name="hid_init",
                                       trainable=learn_init,
                                       regularizable=False)


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units
        else:
            num_outputs = self.num_units*2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        input = T.dot(input, W_in_stacked) + b_stacked

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)

            if self.peepholes:
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)

            cell = forgetgate*cell_previous + ingate*cell_input
            outgate = slice_w(gates, 3)
            outcell = cell
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate

            if self.grad_clipping:
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)

            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]


        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked]
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

class ProjectLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_factors=None,
                 backwards=False,
                 learn_init=False,
                 peepholes=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(ProjectLSTMLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_units/2
        if num_factors:
            self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(self.num_factors, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(spec=init.Uniform(0.1),
                                                   shape=(num_units, ),
                                                   name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(spec=init.Uniform(0.1),
                                                       shape=(num_units, ),
                                                       name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(spec=init.Uniform(0.1),
                                                    shape=(num_units, ),
                                                    name="W_cell_to_outgate")

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, self.num_factors),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=learn_init,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, self.num_factors),
                                       name="hid_init",
                                       trainable=learn_init,
                                       regularizable=False)


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_factors
        else:
            num_outputs = self.num_factors*2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        input = T.dot(input, W_in_stacked) + b_stacked

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)

            if self.peepholes:
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)

            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            outgate = slice_w(gates, 3)
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate

            if self.grad_clipping:
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)

            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    self.W_hid_prj]
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

class LayerNormProjectLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_factors=None,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(LayerNormProjectLSTMLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_units/2
        if num_factors:
            self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(self.num_factors, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        self.W_alpha_in = self.add_param(spec=init.Constant(1.0),
                                         shape=(num_units*4,),
                                         name="W_alpha_in")

        self.W_beta_in = self.add_param(spec=init.Constant(0.0),
                                        shape=(num_units*4,),
                                        name="W_beta_in",
                                        regularizable=False)

        self.W_alpha_hid = self.add_param(spec=init.Constant(1.0),
                                          shape=(num_units*4,),
                                          name="W_alpha_hid")

        self.W_beta_hid = self.add_param(spec=init.Constant(0.0),
                                         shape=(num_units*4,),
                                         name="W_beta_hid",
                                         regularizable=False)

        self.W_alpha_cell = self.add_param(spec=init.Constant(1.0),
                                           shape=(num_units,),
                                           name="W_alpha_cell")

        self.W_beta_cell = self.add_param(spec=init.Constant(0.0),
                                          shape=(num_units,),
                                          name="W_beta_cell",
                                          regularizable=False)

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, self.num_factors),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=learn_init,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, self.num_factors),
                                       name="hid_init",
                                       trainable=learn_init,
                                       regularizable=False)


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_factors
        else:
            num_outputs = self.num_factors*2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        input = T.dot(input, W_in_stacked)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = ln(input_n, self.W_alpha_in, self.W_beta_in)
            gates += ln(T.dot(hid_previous, W_hid_stacked), self.W_alpha_hid, self.W_beta_hid)
            gates += b_stacked

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            outgate = T.nnet.sigmoid(outgate)

            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = ln(cell, self.W_alpha_cell, self.W_beta_cell)

            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    b_stacked,
                    self.W_hid_prj,
                    self.W_alpha_in,
                    self.W_alpha_hid,
                    self.W_alpha_cell,
                    self.W_beta_in,
                    self.W_beta_hid,
                    self.W_beta_cell]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

class CondLayerNormProjectLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_factors,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(CondLayerNormProjectLSTMLayer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units/2, num_units),
                                   name="W_hid_to_{}".format(gate_name)))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate) = add_gate_params('forgetgate')

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate) = add_gate_params('outgate')

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, num_units/2),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units/2),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_inputs, num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, ),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, num_units*4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units*4, ),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Constant(0.),
                                          shape=(num_factors, num_units*4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units*4, ),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_factors, num_units*4),
                                     name="W_beta")
        self.b_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_units*4, ),
                                     name="b_beta",
                                     regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units/2
        else:
            num_outputs = self.num_units + self.num_units/2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        # get ln params
        ln_input = T.dot(input,  self.W_cond_prj)  + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.tanh(ln_input)
        ln_input = T.sum(ln_input*mask.dimshuffle(0, 1, 'x'), axis=1)/T.sum(mask, axis=1, keepdims=True)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + self.b_beta

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        input = T.dot(input, W_in_stacked) + beta
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_ln(input_n, alpha_in)
            gates += sample_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    self.W_hid_prj]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat

class InputCondLayerNormProjectLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 cond_input,
                 mask_input,
                 num_units,
                 num_factors,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     cond_input,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(InputCondLayerNormProjectLSTMLayer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        cond_shape = self.input_shapes[1]
        num_conds = numpy.prod(cond_shape[2:])

        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units/2, num_units),
                                   name="W_hid_to_{}".format(gate_name)))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate) = add_gate_params('forgetgate')

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate) = add_gate_params('outgate')

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, num_units/2),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units/2),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_conds, num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, ),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, num_units*4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units*4, ),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Constant(0.),
                                          shape=(num_factors, num_units*4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units*4, ),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_factors, num_units*4),
                                     name="W_beta")
        self.b_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_units*4, ),
                                     name="b_beta",
                                     regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units/2
        else:
            num_outputs = self.num_units + self.num_units/2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        cond_data = inputs[1]
        mask = inputs[2]

        # get ln params
        ln_input = T.dot(cond_data,  self.W_cond_prj)  + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.tanh(ln_input)
        ln_input = T.sum(ln_input*mask.dimshuffle(0, 1, 'x'), axis=1)/T.sum(mask, axis=1, keepdims=True)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + self.b_beta

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        input = T.dot(input, W_in_stacked) + beta
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_ln(input_n, alpha_in)
            gates += sample_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    self.W_hid_prj]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat

class MaxCondLayerNormProjectLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_factors,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(MaxCondLayerNormProjectLSTMLayer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units/2, num_units),
                                   name="W_hid_to_{}".format(gate_name)))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate) = add_gate_params('forgetgate')

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate) = add_gate_params('outgate')

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, num_units/2),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units/2),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_inputs, num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, ),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, num_units*4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units*4, ),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Constant(0.),
                                          shape=(num_factors, num_units*4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units*4, ),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_factors, num_units*4),
                                     name="W_beta")
        self.b_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_units*4, ),
                                     name="b_beta",
                                     regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units/2
        else:
            num_outputs = self.num_units + self.num_units/2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        # get ln params
        ln_input = T.dot(input,  self.W_cond_prj)  + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.nnet.relu(ln_input)
        ln_input = T.max(ln_input, axis=1)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + self.b_beta

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        input = T.dot(input, W_in_stacked) + beta
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_ln(input_n, alpha_in)
            gates += sample_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    self.W_hid_prj]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat

class MaxInputCondLayerNormProjectLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 cond_input,
                 mask_input,
                 num_units,
                 num_factors,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     cond_input,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(MaxInputCondLayerNormProjectLSTMLayer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        cond_shape = self.input_shapes[1]
        num_conds = numpy.prod(cond_shape[2:])

        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units/2, num_units),
                                   name="W_hid_to_{}".format(gate_name)))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate) = add_gate_params('forgetgate')

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate) = add_gate_params('outgate')

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, num_units/2),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units/2),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_conds, num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, ),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, num_units*4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units*4, ),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Constant(0.),
                                          shape=(num_factors, num_units*4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units*4, ),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_factors, num_units*4),
                                     name="W_beta")
        self.b_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_units*4, ),
                                     name="b_beta",
                                     regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units/2
        else:
            num_outputs = self.num_units + self.num_units/2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        cond_data = inputs[1]
        mask = inputs[2]

        # get ln params
        ln_input = T.dot(cond_data,  self.W_cond_prj)  + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.nnet.relu(ln_input)
        ln_input = T.max(ln_input, axis=1)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + self.b_beta

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        input = T.dot(input, W_in_stacked) + beta
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_ln(input_n, alpha_in)
            gates += sample_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    self.W_hid_prj]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat


#################
# Fixed Version #
#################
class ProjectLSTM_v0_1_Layer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_prjs,
                 backwards=False,
                 learn_init=False,
                 peepholes=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(ProjectLSTM_v0_1_Layer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_prjs = num_prjs

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(self.num_prjs, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(spec=init.Uniform(0.1),
                                                   shape=(num_units, ),
                                                   name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(spec=init.Uniform(0.1),
                                                       shape=(num_units, ),
                                                       name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(spec=init.Uniform(0.1),
                                                    shape=(num_units, ),
                                                    name="W_cell_to_outgate")

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, self.num_prjs),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=learn_init,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, self.num_prjs),
                                       name="hid_init",
                                       trainable=learn_init,
                                       regularizable=False)


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_prjs
        else:
            num_outputs = self.num_prjs + self.num_units

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        input = T.dot(input, W_in_stacked) + b_stacked

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)

            if self.peepholes:
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)

            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            outgate = slice_w(gates, 3)
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate

            if self.grad_clipping:
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)

            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    self.W_hid_prj]
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)


class LayerNormProjectLSTM_v0_1_Layer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_prjs,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(LayerNormProjectLSTM_v0_1_Layer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_prjs = num_prjs

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(self.num_prjs, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        self.W_alpha_in = self.add_param(spec=init.Constant(1.0),
                                         shape=(num_units*4,),
                                         name="W_alpha_in")

        self.W_alpha_hid = self.add_param(spec=init.Constant(1.0),
                                          shape=(num_units*4,),
                                          name="W_alpha_hid")

        self.W_alpha_cell = self.add_param(spec=init.Constant(1.0),
                                           shape=(num_units,),
                                           name="W_alpha_cell")

        self.W_beta_cell = self.add_param(spec=init.Constant(0.0),
                                          shape=(num_units,),
                                          name="W_beta_cell",
                                          regularizable=False)

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, self.num_prjs),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=learn_init,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, self.num_prjs),
                                       name="hid_init",
                                       trainable=learn_init,
                                       regularizable=False)


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_prjs
        else:
            num_outputs = self.num_prjs + self.num_units

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        input = T.dot(input, W_in_stacked)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = ln(input_n, self.W_alpha_in)
            gates += ln(T.dot(hid_previous, W_hid_stacked), self.W_alpha_hid)
            gates += b_stacked

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            outgate = T.nnet.sigmoid(outgate)

            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = ln(cell, self.W_alpha_cell, self.W_beta_cell)

            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    b_stacked,
                    self.W_hid_prj,
                    self.W_alpha_in,
                    self.W_alpha_hid,
                    self.W_alpha_cell,
                    self.W_beta_cell]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)


class LayerNormProjectLSTM_v0_2_Layer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_prjs,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(LayerNormProjectLSTM_v0_2_Layer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_prjs = num_prjs

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(self.num_prjs, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        self.W_alpha_in = self.add_param(spec=init.Constant(1.0),
                                         shape=(num_units*4,),
                                         name="W_alpha_in")

        self.W_alpha_hid = self.add_param(spec=init.Constant(1.0),
                                          shape=(num_units*4,),
                                          name="W_alpha_hid")

        self.W_alpha_cell = self.add_param(spec=init.Constant(1.0),
                                           shape=(num_units,),
                                           name="W_alpha_cell")

        self.W_beta_cell = self.add_param(spec=init.Constant(0.0),
                                          shape=(num_units,),
                                          name="W_beta_cell",
                                          regularizable=False)

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, self.num_prjs),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=learn_init,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, self.num_prjs),
                                       name="hid_init",
                                       trainable=learn_init,
                                       regularizable=False)


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_prjs
        else:
            num_outputs = self.num_prjs + self.num_units

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        input = T.dot(input, W_in_stacked)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = gate_ln(input_n, self.W_alpha_in)
            gates += gate_ln(T.dot(hid_previous, W_hid_stacked), self.W_alpha_hid)
            gates += b_stacked

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            outgate = T.nnet.sigmoid(outgate)

            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = ln(cell, self.W_alpha_cell, self.W_beta_cell)

            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    b_stacked,
                    self.W_hid_prj,
                    self.W_alpha_in,
                    self.W_alpha_hid,
                    self.W_alpha_cell,
                    self.W_beta_cell]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)


class CondLayerNormProjectLSTM_v0_1_Layer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_prjs,
                 num_factors=None,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(CondLayerNormProjectLSTM_v0_1_Layer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_prjs = num_prjs

        if num_factors:
            self.num_factors = num_factors
        else:
            self.num_factors = num_units/2

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(self.num_prjs, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, self.num_prjs),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, self.num_prjs),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_inputs, self.num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(self.num_factors, ),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Uniform(0.01),
                                         shape=(self.num_factors, num_units*4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units*4, ),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Uniform(0.01),
                                          shape=(self.num_factors, num_units*4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units*4, ),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Uniform(0.01),
                                     shape=(self.num_factors, num_units*4),
                                     name="W_beta")

        #### cell  ####
        self.W_alpha_cell = self.add_param(spec=init.Constant(1.0),
                                           shape=(num_units,),
                                           name="W_alpha_cell")

        self.W_beta_cell = self.add_param(spec=init.Constant(0.0),
                                          shape=(num_units,),
                                          name="num_factors",
                                          regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_prjs
        else:
            num_outputs = self.num_prjs + self.num_units

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        # concat params
        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        # get ln params
        ln_input = T.dot(input,  self.W_cond_prj) + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.tanh(ln_input)
        ln_input = T.sum(ln_input*mask.dimshuffle(0, 1, 'x'), axis=1)/T.sum(mask, axis=1, keepdims=True)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + b_stacked

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        input = T.dot(input, W_in_stacked)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_ln(input_n, alpha_in)
            gates += sample_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)
            gates += beta

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = ln(cell, self.W_alpha_cell, self.W_beta_cell)
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    beta,
                    self.W_hid_prj,
                    self.W_alpha_cell,
                    self.W_beta_cell]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat


class CondLayerNormProjectLSTM_v0_2_Layer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_prjs,
                 num_factors=None,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(CondLayerNormProjectLSTM_v0_2_Layer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_prjs = num_prjs

        if num_factors:
            self.num_factors = num_factors
        else:
            self.num_factors = num_units / 2

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(self.num_prjs, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        #### hidden projection ####
        self.W_hid_prj = self.add_param(spec=init.Orthogonal(),
                                        shape=(num_units, self.num_prjs),
                                        name="W_hid_prj")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, self.num_prjs),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_inputs, self.num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(self.num_factors,),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Uniform(0.01),
                                         shape=(self.num_factors, num_units * 4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units * 4,),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Uniform(0.01),
                                          shape=(self.num_factors, num_units * 4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units * 4,),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Uniform(0.01),
                                     shape=(self.num_factors, num_units * 4),
                                     name="W_beta")

        #### cell  ####
        self.W_alpha_cell = self.add_param(spec=init.Constant(1.0),
                                           shape=(num_units,),
                                           name="W_alpha_cell")

        self.W_beta_cell = self.add_param(spec=init.Constant(0.0),
                                          shape=(num_units,),
                                          name="num_factors",
                                          regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_prjs
        else:
            num_outputs = self.num_prjs + self.num_units

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        # concat params
        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        # get ln params
        ln_input = T.dot(input, self.W_cond_prj) + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.tanh(ln_input)
        ln_input = T.sum(ln_input * mask.dimshuffle(0, 1, 'x'), axis=1) / T.sum(mask, axis=1, keepdims=True)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + b_stacked

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        input = T.dot(input, W_in_stacked)

        def slice_w(x, n):
            return x[:, n * self.num_units:(n + 1) * self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_gate_ln(input_n, alpha_in)
            gates += sample_gate_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)
            gates += beta

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate * cell_previous + ingate * cell_input
            outcell = ln(cell, self.W_alpha_cell, self.W_beta_cell)
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate * T.tanh(outcell)
            hid = T.dot(hid, self.W_hid_prj)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    beta,
                    self.W_hid_prj,
                    self.W_alpha_cell,
                    self.W_beta_cell]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat


#################
# No projection #
#################

class LayerNormLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(LayerNormLSTMLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name, const_bias=0.0):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(const_bias),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate', const_bias=1.0)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')

        self.W_alpha_in = self.add_param(spec=init.Constant(1.0),
                                         shape=(num_units*4,),
                                         name="W_alpha_in")

        self.W_beta_in = self.add_param(spec=init.Constant(0.0),
                                        shape=(num_units*4,),
                                        name="W_beta_in",
                                        regularizable=False)

        self.W_alpha_hid = self.add_param(spec=init.Constant(1.0),
                                          shape=(num_units*4,),
                                          name="W_alpha_hid")

        self.W_beta_hid = self.add_param(spec=init.Constant(0.0),
                                         shape=(num_units*4,),
                                         name="W_beta_hid",
                                         regularizable=False)

        self.W_alpha_cell = self.add_param(spec=init.Constant(1.0),
                                           shape=(num_units,),
                                           name="W_alpha_cell")

        self.W_beta_cell = self.add_param(spec=init.Constant(0.0),
                                          shape=(num_units,),
                                          name="W_beta_cell",
                                          regularizable=False)

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=learn_init,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units),
                                       name="hid_init",
                                       trainable=learn_init,
                                       regularizable=False)


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units
        else:
            num_outputs = self.num_units*2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        input = T.dot(input, W_in_stacked)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = ln(input_n, self.W_alpha_in, self.W_beta_in)
            gates += ln(T.dot(hid_previous, W_hid_stacked), self.W_alpha_hid, self.W_beta_hid)
            gates += b_stacked

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            outgate = T.nnet.sigmoid(outgate)

            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = ln(cell, self.W_alpha_cell, self.W_beta_cell)

            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            hid = outgate*T.tanh(outcell)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    b_stacked,
                    self.W_alpha_in,
                    self.W_alpha_hid,
                    self.W_alpha_cell,
                    self.W_beta_in,
                    self.W_beta_hid,
                    self.W_beta_cell]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

class CondLayerNormLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 mask_input,
                 num_units,
                 num_factors,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(CondLayerNormLSTMLayer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate) = add_gate_params('forgetgate')

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate) = add_gate_params('outgate')


        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_inputs, num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, ),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, num_units*4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units*4, ),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Constant(0.),
                                          shape=(num_factors, num_units*4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units*4, ),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_factors, num_units*4),
                                     name="W_beta")
        self.b_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_units*4, ),
                                     name="b_beta",
                                     regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units
        else:
            num_outputs = self.num_units*2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        mask = inputs[1]

        # get ln params
        ln_input = T.dot(input,  self.W_cond_prj)  + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.tanh(ln_input)
        ln_input = T.sum(ln_input*mask.dimshuffle(0, 1, 'x'), axis=1)/T.sum(mask, axis=1, keepdims=True)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + self.b_beta

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        input = T.dot(input, W_in_stacked)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_ln(input_n, alpha_in)
            gates += sample_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)
            gates += beta

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    beta]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat

class InputCondLayerNormLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 cond_input,
                 mask_input,
                 num_units,
                 num_factors,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming,
                     cond_input,
                     mask_input]

        cell_init = init.Constant(0.)
        hid_init = init.Constant(0.)

        super(InputCondLayerNormLSTMLayer, self).__init__(incomings, **kwargs)

        self.sample_feat = None

        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        self.num_factors = num_factors

        input_shape = self.input_shapes[0]
        num_inputs = numpy.prod(input_shape[2:])

        cond_shape = self.input_shapes[1]
        num_conds = numpy.prod(cond_shape[2:])

        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)))

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate) = add_gate_params('ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate) = add_gate_params('forgetgate')

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell) = add_gate_params('cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate) = add_gate_params('outgate')

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.add_param(spec=cell_init,
                                        shape=(1, num_units),
                                        name="cell_init",
                                        trainable=False,
                                        regularizable=False)
        self.hid_init = self.add_param(spec=hid_init,
                                       shape=(1, num_units),
                                       name="hid_init",
                                       trainable=False,
                                       regularizable=False)

        #### conditional mapping ####
        self.W_cond_prj = self.add_param(spec=init.Orthogonal(),
                                         shape=(num_conds, num_factors),
                                         name="W_cond_prj")
        self.b_cond_prj = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, ),
                                         name="b_cond_prj",
                                         regularizable=False)

        #### input layer norm ####
        self.W_alpha_in = self.add_param(spec=init.Constant(0.),
                                         shape=(num_factors, num_units*4),
                                         name="W_alpha_in")
        self.b_alpha_in = self.add_param(spec=init.Constant(1.),
                                         shape=(num_units*4, ),
                                         name="b_alpha_in")

        #### hidden layer norm ####
        self.W_alpha_hid = self.add_param(spec=init.Constant(0.),
                                          shape=(num_factors, num_units*4),
                                          name="W_alpha_hid")
        self.b_alpha_hid = self.add_param(spec=init.Constant(1.),
                                          shape=(num_units*4, ),
                                          name="b_alpha_hid")

        #### beta  for all ####
        self.W_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_factors, num_units*4),
                                     name="W_beta")
        self.b_beta = self.add_param(spec=init.Constant(0.),
                                     shape=(num_units*4, ),
                                     name="b_beta",
                                     regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_units
        else:
            num_outputs = self.num_units*2

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        cond_data = inputs[1]
        mask = inputs[2]

        # get ln params
        ln_input = T.dot(cond_data,  self.W_cond_prj)  + self.b_cond_prj.dimshuffle('x', 'x', 0)
        ln_input = T.tanh(ln_input)
        ln_input = T.sum(ln_input*mask.dimshuffle(0, 1, 'x'), axis=1)/T.sum(mask, axis=1, keepdims=True)

        self.sample_feat = ln_input

        alpha_in = T.dot(ln_input, self.W_alpha_in) + self.b_alpha_in
        alpha_hid = T.dot(ln_input, self.W_alpha_hid) + self.b_alpha_hid
        beta = T.dot(ln_input, self.W_beta) + self.b_beta

        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input.shape

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        input = T.dot(input, W_in_stacked)
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = sample_ln(input_n, alpha_in)
            gates += sample_ln(T.dot(hid_previous, W_hid_stacked), alpha_hid)
            gates += beta

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(ingate,
                                                   -self.grad_clipping,
                                                   self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(forgetgate,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                cell_input = theano.gradient.grad_clip(cell_input,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)
            cell = forgetgate*cell_previous + ingate*cell_input
            outcell = cell
            if self.grad_clipping:
                outcell = theano.gradient.grad_clip(outcell,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(outcell)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n,
                             cell_previous,
                             hid_previous,
                             *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        sequences = [input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked,
                    alpha_in,
                    alpha_hid,
                    beta]

        cell_out, hid_out = theano.scan(fn=step_fun,
                                        sequences=sequences,
                                        outputs_info=[cell_init,
                                                      hid_init],
                                        go_backwards=self.backwards,
                                        truncate_gradient=self.gradient_steps,
                                        non_sequences=non_seqs,
                                        strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                cell_out = cell_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

    def get_sample_feat(self):
        return self.sample_feat

