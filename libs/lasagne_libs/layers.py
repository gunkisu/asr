import numpy
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.layers import Layer, MergeLayer, ConcatLayer
from lasagne import nonlinearities, init
from lasagne.layers import get_output
from lasagne.utils import unroll_scan
from lasagne.random import get_rng

from lasagne.layers import DenseLayer, ReshapeLayer, reshape

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

class Gate(object):
    def __init__(self,
                 W_in=init.Orthogonal(0.1),
                 W_hid=init.Orthogonal(0.1),
                 W_cell=init.Uniform(0.1),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

class SequenceLayerNormLayer(Layer):
    def __init__(self,
                 incoming,
                 **kwargs):

        super(SequenceLayerNormLayer, self).__init__(incoming, **kwargs)

        num_feats = self.input_shape[-1]

        self.alpha = self.add_param(init.Constant(1.), (num_feats,), name="alpha")
        self.beta = self.add_param(init.Constant(0.), (num_feats,), name="beta", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        output = (input - T.mean(input, axis=-1, keepdims=True))/T.sqrt(T.var(input, axis=-1, keepdims=True) + eps)
        output = self.alpha[None, None, :]*output + self.beta[None, None, :]
        return output

class SequenceDenseLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_outputs,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.),
                 mask_input=None,
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):

        incomings = [incoming,]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1

        self.num_outputs = num_outputs

        super(SequenceDenseLayer, self).__init__(incomings, **kwargs)

        num_inputs = self.input_shapes[0][-1]

        self.W = self.add_param(W, (num_inputs, num_outputs), name="W")

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_outputs,), name="b", regularizable=False)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearities.softmax


    def get_output_shape_for(self, input_shapes):
        num_samples = input_shapes[0][0]
        num_steps = input_shapes[0][1]
        return (num_samples, num_steps, self.num_outputs)

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index>0:
            mask = inputs[1]

        # dense operation
        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b[None, None, :]

        if self.nonlinearity==nonlinearities.softmax:
            # softmax operation for probability
            activation = activation - T.max(activation, axis=-1, keepdims=True)
            activation = T.exp(activation)
            if mask:
                activation = activation*mask[:, :, None]
            output = activation/(T.sum(activation, axis=-1, keepdims=True) + eps)
        else:
            output = self.nonlinearity(activation)
        return output

class LSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_units,
                 ingate=Gate(),
                 forgetgate=Gate(init.Constant(1.)),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 dropout_ratio=0.2,
                 use_layer_norm=True,
                 weight_noise=0.0,
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 mask_input=None,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        self.hid_init_incoming_index = -1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        self.cell_init_incoming_index = -1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incomings, **kwargs)

        # for dropout
        self.binomial = RandomStreams(get_rng().randint(1, 2147462579)).binomial
        self.p = dropout_ratio

        # for layer norm
        self.use_layer_norm = use_layer_norm

        # for weight noise
        self.normal = RandomStreams(get_rng().randint(1, 2147462579)).normal

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.weight_noise = weight_noise
        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        if unroll_scan and gradient_steps != -1:
            raise ValueError("Gradient steps must be -1 when unroll_scan is true.")

        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        #### weight init ####
        num_inputs = numpy.prod(input_shape[2:])
        def add_gate_params(gate, gate_name):
            return (self.add_param(spec=gate.W_in,
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=gate.W_hid,
                                   shape=(num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=gate.b,
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        if self.use_layer_norm:
            self.alpha_ingate = self.add_param(spec=init.Constant(1.),
                                               shape=(num_units,),
                                               name="alpha_ingate")
            self.beta_ingate =  self.add_param(spec=init.Constant(0.),
                                               shape=(num_units,),
                                               name="beta_ingate",
                                               regularizable=False)
        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate, 'forgetgate')
        if self.use_layer_norm:
            self.alpha_forgetgate = self.add_param(spec=init.Constant(1.),
                                                   shape=(num_units,),
                                                   name="alpha_forgetgate")
            self.beta_forgetgate =  self.add_param(spec=init.Constant(0.),
                                                   shape=(num_units,),
                                                   name="beta_forgetgate",
                                                   regularizable=False)

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        if self.use_layer_norm:
            self.alpha_cell = self.add_param(spec=init.Constant(1.),
                                             shape=(num_units,),
                                             name="alpha_cell")
            self.beta_cell =  self.add_param(spec=init.Constant(0.),
                                             shape=(num_units,),
                                             name="beta_cell",
                                             regularizable=False)

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')
        if self.use_layer_norm:
            self.alpha_outgate = self.add_param(spec=init.Constant(1.),
                                                shape=(num_units,),
                                                name="alpha_outgate")
            self.beta_outgate =  self.add_param(spec=init.Constant(0.),
                                                shape=(num_units,),
                                                name="beta_outgate",
                                                regularizable=False)

        #### peepholes ####
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(spec=ingate.W_cell,
                                                   shape=(num_units, ),
                                                   name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(spec=forgetgate.W_cell,
                                                       shape=(num_units, ),
                                                       name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(spec=outgate.W_cell,
                                                    shape=(num_units, ),
                                                    name="W_cell_to_outgate")

        #### out cell ####
        if self.use_layer_norm:
            self.alpha_outcell = self.add_param(spec=init.Constant(1.),
                                                shape=(num_units,),
                                                name="alpha_outcell")
            self.beta_outcell =  self.add_param(spec=init.Constant(0.),
                                                shape=(num_units,),
                                                name="beta_outcell",
                                                regularizable=False)

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def layer_norm(self, input, alpha, beta):
        output = (input - T.mean(input, axis=1, keepdims=True))/(T.sqrt(T.var(input, axis=1, keepdims=True)) + eps)
        output = alpha[None, :]*output + beta[None, :]
        return output

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
        input = inputs[0]

        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        hid_init = None
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        cell_init = None
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        #### input ####
        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        #### hidden ####
        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        #### bias ####
        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        #### weight noise ####
        if self.weight_noise>0 and deterministic is True:
            W_in_stacked += self.normal(size=W_in_stacked.shape,
                                        std=self.weight_noise)
            W_hid_stacked += self.normal(size=W_hid_stacked.shape,
                                         std=self.weight_noise)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        #### set dropout mask ####
        if deterministic:
            self.using_dropout = False
        else:
            self.using_dropout = True
        cell_mask = self.binomial((num_batch, self.num_units),
                                  p=T.constant(1) - self.p,
                                  dtype=floatX)

        input = T.dot(input, W_in_stacked) + b_stacked
        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            if self.use_layer_norm:
                ingate = self.layer_norm(input=ingate,
                                         alpha=self.alpha_ingate,
                                         beta=self.beta_ingate)
                forgetgate = self.layer_norm(input=forgetgate,
                                             alpha=self.alpha_forgetgate,
                                             beta=self.beta_forgetgate)
                cell_input = self.layer_norm(input=cell_input,
                                             alpha=self.alpha_cell,
                                             beta=self.beta_cell)
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

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            if self.using_dropout==False or self.p == 0:
                cell_input = cell_input
            else:
                one = T.constant(1)
                retain_prob = one - self.p
                cell_input /= retain_prob
                cell_input = cell_input*cell_mask

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate

            if self.use_layer_norm:
                outgate = self.layer_norm(input=outgate,
                                          alpha=self.alpha_outgate,
                                          beta=self.beta_outgate)
            if self.grad_clipping:
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
                cell = theano.gradient.grad_clip(cell,
                                                 -self.grad_clipping,
                                                 self.grad_clipping)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            if self.use_layer_norm:
                norm_cell = self.layer_norm(input=cell,
                                            alpha=self.alpha_outcell,
                                            beta=self.beta_outcell)
                hid = outgate * self.nonlinearity(norm_cell)
            else:
                hid = outgate * self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            hid_init = T.dot(ones, self.hid_init)

        non_seqs = [cell_mask, W_hid_stacked]
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        if self.use_layer_norm:
            non_seqs +=[self.alpha_ingate,
                        self.alpha_forgetgate,
                        self.alpha_cell,
                        self.alpha_outgate,
                        self.alpha_outcell,
                        self.beta_ingate,
                        self.beta_forgetgate,
                        self.beta_cell,
                        self.beta_outgate,
                        self.beta_outcell,]

        if self.unroll_scan:
            input_shape = self.input_shapes[0]
            cell_out, hid_out = unroll_scan(fn=step_fun,
                                            sequences=sequences,
                                            outputs_info=[cell_init, hid_init],
                                            non_sequences=non_seqs,
                                            n_steps=input_shape[1])
        else:
            cell_out, hid_out = theano.scan(fn=step_fun,
                                            sequences=sequences,
                                            outputs_info=[cell_init, hid_init],
                                            go_backwards=self.backwards,
                                            truncate_gradient=self.gradient_steps,
                                            non_sequences=non_seqs,
                                            strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                cell_out = cell_out.dimshuffle(1, 0, 2)

                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)

class LSTMPLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_prj,
                 num_units,
                 ingate=Gate(),
                 forgetgate=Gate(b=init.Constant(1.)),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 dropout_ratio=0.2,
                 weight_noise=0.0,
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 mask_input=None,
                 only_return_final=False,
                 only_return_hidden=True,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        self.hid_init_incoming_index = -1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        self.cell_init_incoming_index = -1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(LSTMPLayer, self).__init__(incomings, **kwargs)

        # for dropout
        self.binomial = RandomStreams(get_rng().randint(1, 2147462579)).binomial
        self.p = dropout_ratio

        # for weight noise
        self.normal = RandomStreams(get_rng().randint(1, 2147462579)).normal

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.weight_noise = weight_noise
        self.learn_init = learn_init
        self.num_prj = num_prj
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.only_return_final = only_return_final
        self.only_return_hidden = only_return_hidden

        if unroll_scan and gradient_steps != -1:
            raise ValueError("Gradient steps must be -1 when unroll_scan is true.")

        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        #### weight init ####
        num_inputs = numpy.prod(input_shape[2:])
        def add_gate_params(gate, gate_name):
            return (self.add_param(spec=gate.W_in,
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=gate.W_hid,
                                   shape=(num_prj, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=gate.b,
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        #### ingate ####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        #### forgetgate ####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate, 'forgetgate')

        #### cell ####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        #### outgate ####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        #### peepholes ####
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(spec=ingate.W_cell,
                                                   shape=(num_units, ),
                                                   name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(spec=forgetgate.W_cell,
                                                       shape=(num_units, ),
                                                       name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(spec=outgate.W_cell,
                                                    shape=(num_units, ),
                                                    name="W_cell_to_outgate")

        #### hidden projection ####
        self.W_hid_projection = self.add_param(spec=init.Orthogonal(),
                                               shape=(num_units, num_prj),
                                               name="W_cell_to_outgate")


        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, num_prj), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_hidden:
            num_outputs = self.num_prj
        else:
            num_outputs = self.num_prj + self.num_units

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input = inputs[0]

        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        hid_init = None
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        cell_init = None
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        #### input ####
        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        #### hidden ####
        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        #### bias ####
        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        #### weight noise ####
        if self.weight_noise>0 and deterministic is True:
            W_in_stacked += self.normal(size=W_in_stacked.shape,
                                        std=self.weight_noise)
            W_hid_stacked += self.normal(size=W_hid_stacked.shape,
                                         std=self.weight_noise)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        #### set dropout mask ####
        if deterministic:
            self.using_dropout = False
        else:
            self.using_dropout = True
        cell_mask = self.binomial((num_batch, self.num_units),
                                  p=T.constant(1) - self.p,
                                  dtype=floatX)

        input = T.dot(input, W_in_stacked) + b_stacked
        def step(input_n,
                 cell_previous,
                 hid_previous,
                 *args):
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

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

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            if self.using_dropout==False or self.p == 0:
                cell_input = cell_input
            else:
                one = T.constant(1)
                retain_prob = one - self.p
                cell_input /= retain_prob
                cell_input = cell_input*cell_mask

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate

            if self.grad_clipping:
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)
                cell = theano.gradient.grad_clip(cell,
                                                 -self.grad_clipping,
                                                 self.grad_clipping)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate * self.nonlinearity(cell)
            hid = T.dot(hid, self.W_hid_projection)
            return [cell, hid]

        def step_masked(input_n,
                        mask_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            hid_init = T.dot(ones, self.hid_init)

        non_seqs = [cell_mask, W_hid_stacked]
        non_seqs += [self.W_hid_projection, ]
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        if self.unroll_scan:
            input_shape = self.input_shapes[0]
            cell_out, hid_out = unroll_scan(fn=step_fun,
                                            sequences=sequences,
                                            outputs_info=[cell_init, hid_init],
                                            non_sequences=non_seqs,
                                            n_steps=input_shape[1])
        else:
            cell_out, hid_out = theano.scan(fn=step_fun,
                                            sequences=sequences,
                                            outputs_info=[cell_init, hid_init],
                                            go_backwards=self.backwards,
                                            truncate_gradient=self.gradient_steps,
                                            non_sequences=non_seqs,
                                            strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        if self.only_return_hidden:
            return hid_out
        else:
            if self.only_return_final:
                cell_out = cell_out[-1]
            else:
                cell_out = cell_out.dimshuffle(1, 0, 2)

                if self.backwards:
                    cell_out = cell_out[:, ::-1]

            return T.concatenate([hid_out, cell_out], axis=-1)



