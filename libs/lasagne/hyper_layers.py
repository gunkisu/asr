import theano
import numpy
from theano import tensor as T
from lasagne import init, nonlinearities
from lasagne.layers import Layer, MergeLayer, Gate, ConcatLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from lasagne.utils import unroll_scan
from lasagne.layers import get_output
floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

class ScaleHyperLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_inner_units,
                 num_inner_factor_units,
                 num_outer_units,
                 inner_ingate=Gate(),
                 inner_forgetgate=Gate(b=init.Constant(1.)),
                 inner_cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 inner_outgate=Gate(),
                 outer_ingate=Gate(),
                 outer_forgetgate=Gate(b=init.Constant(1.)),
                 outer_cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outer_outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 inner_cell_init=init.Constant(0.),
                 inner_hid_init=init.Constant(0.),
                 outer_cell_init=init.Constant(0.),
                 outer_hid_init=init.Constant(0.),
                 dropout_ratio=0.2,
                 use_layer_norm=True,
                 weight_noise=0.0,
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        incomings = [incoming]

        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        self.inner_hid_init_incoming_index = -1
        if isinstance(inner_hid_init, Layer):
            incomings.append(inner_hid_init)
            self.inner_hid_init_incoming_index = len(incomings)-1

        self.inner_cell_init_incoming_index = -1
        if isinstance(inner_cell_init, Layer):
            incomings.append(inner_cell_init)
            self.inner_cell_init_incoming_index = len(incomings)-1

        self.outer_hid_init_incoming_index = -1
        if isinstance(outer_hid_init, Layer):
            incomings.append(outer_hid_init)
            self.outer_hid_init_incoming_index = len(incomings)-1

        self.outer_cell_init_incoming_index = -1
        if isinstance(outer_cell_init, Layer):
            incomings.append(outer_cell_init)
            self.outer_cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(ScaleHyperLSTMLayer, self).__init__(incomings, **kwargs)

        # for dropout
        self.binomial = RandomStreams(get_rng().randint(1, 2147462579)).binomial
        self.p = dropout_ratio

        # for layer norm
        self.use_layer_norm = use_layer_norm

        # for weight noise
        self.weight_noise = weight_noise
        self.normal = RandomStreams(get_rng().randint(1, 2147462579)).normal

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_inner_units = num_inner_units
        self.num_inner_factor_units = num_inner_factor_units
        self.num_outer_units = num_outer_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError("Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = numpy.prod(input_shape[2:])

        ##############
        # inner loop #
        ##############
        def add_inner_gate_params(gate, gate_name):
            return (self.add_param(gate.W_in,
                                   shape=(num_inputs, num_inner_units),
                                   name="W_inner_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid,
                                   shape=(num_inner_units, num_inner_units),
                                   name="W_inner_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b,
                                   shape=(num_inner_units,),
                                   name="b_inner_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        ####ingate####
        (self.W_inner_in_to_ingate,
         self.W_inner_hid_to_ingate,
         self.b_inner_ingate,
         self.nonlinearity_inner_ingate) = add_inner_gate_params(inner_ingate, 'ingate')

        ####forgetgate#####
        (self.W_inner_in_to_forgetgate,
         self.W_inner_hid_to_forgetgate,
         self.b_inner_forgetgate,
         self.nonlinearity_inner_forgetgate) = add_inner_gate_params(inner_forgetgate, 'forgetgate')

        ####cell#####
        (self.W_inner_in_to_cell,
         self.W_inner_hid_to_cell,
         self.b_inner_cell,
         self.nonlinearity_inner_cell) = add_inner_gate_params(inner_cell, 'cell')

        ####outgate#####
        (self.W_inner_in_to_outgate,
         self.W_inner_hid_to_outgate,
         self.b_inner_outgate,
         self.nonlinearity_inner_outgate) = add_inner_gate_params(inner_outgate, 'outgate')

        ####peepholes#####
        if self.peepholes:
            self.W_inner_cell_to_ingate = self.add_param(inner_ingate.W_cell,
                                                         shape=(num_inner_units, ),
                                                         name="W_inner_cell_to_ingate")

            self.W_inner_cell_to_forgetgate = self.add_param(inner_forgetgate.W_cell,
                                                             shape=(num_inner_units, ),
                                                             name="W_inner_cell_to_forgetgate")

            self.W_inner_cell_to_outgate = self.add_param(inner_outgate.W_cell,
                                                          shape=(num_inner_units, ),
                                                          name="W_inner_cell_to_outgate")

        ####layernorm#####
        if self.use_layer_norm:
            self.alpha_inner_gate = self.add_param(init.Constant(1.),
                                                   shape=(num_inner_units*4,),
                                                   name="alpha_inner_gate")
            self.beta_inner_gate =  self.add_param(init.Constant(0.),
                                                   shape=(num_inner_units*4,),
                                                   name="beta_inner_gate",
                                                   regularizable=False)

            self.alpha_inner_cell = self.add_param(init.Constant(1.),
                                                   shape=(num_inner_units,),
                                                   name="alpha_inner_cell")
            self.beta_inner_cell =  self.add_param(init.Constant(0.),
                                                   shape=(num_inner_units,),
                                                   name="beta_inner_cell",
                                                   regularizable=False)

        ####init_cell####
        if isinstance(inner_cell_init, Layer):
            self.inner_cell_init = inner_cell_init
        else:
            self.inner_cell_init = self.add_param(inner_cell_init,
                                                  shape=(1, num_inner_units),
                                                  name="inner_cell_init",
                                                  trainable=learn_init,
                                                  regularizable=False)

        ####init_hidden####
        if isinstance(inner_hid_init, Layer):
            self.inner_hid_init = inner_hid_init
        else:
            self.inner_hid_init = self.add_param(inner_hid_init,
                                                 shape=(1, num_inner_units),
                                                 name="inner_hid_init",
                                                 trainable=learn_init,
                                                 regularizable=False)

        ###################
        # inner-to-factor #
        ###################
        def add_inner2fact_params(gate, gate_name):
            return (self.add_param(gate.W_hid,
                                   shape=(num_inner_units, num_inner_factor_units),
                                   name="V0_in_to_{}".format(gate_name)),
                    self.add_param(gate.b,
                                   shape=(num_inner_factor_units,),
                                   name="b0_in_to_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(gate.W_hid,
                                   shape=(num_inner_units, num_inner_factor_units),
                                   name="V0_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b,
                                   shape=(num_inner_factor_units,),
                                   name="b0_hid_to_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(gate.W_hid,
                                   shape=(num_inner_units, num_inner_factor_units),
                                   name="V0_bias_{}".format(gate_name)))

        ####ingate####
        (self.V0_in_to_ingate,
         self.b0_in_to_ingate,
         self.V0_hid_to_ingate,
         self.b0_hid_to_ingate,
         self.V0_bias_ingate) = add_inner2fact_params(outer_ingate, 'ingate')

        ####forgetgate####
        (self.V0_in_to_forgetgate,
         self.b0_in_to_forgetgate,
         self.V0_hid_to_forgetgate,
         self.b0_hid_to_forgetgate,
         self.V0_bias_forgetgate) = add_inner2fact_params(outer_forgetgate, 'forgetgate')

        ####cell####
        (self.V0_in_to_cell,
         self.b0_in_to_cell,
         self.V0_hid_to_cell,
         self.b0_hid_to_cell,
         self.V0_bias_cell) = add_inner2fact_params(outer_cell, 'cell')

        ####outgate####
        (self.V0_in_to_outgate,
         self.b0_in_to_outgate,
         self.V0_hid_to_outgate,
         self.b0_hid_to_outgate,
         self.V0_bias_outgate) = add_inner2fact_params(outer_outgate, 'outgate')

        ###################
        # factor-to-outer #
        ###################
        def add_fact2outer_params(gate, gate_name):
            return (self.add_param(gate.W_hid,
                                   shape=(num_inner_factor_units, num_outer_units),
                                   name="V1_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid,
                                   shape=(num_inner_factor_units, num_outer_units),
                                   name="V1_hid_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid,
                                   shape=(num_inner_factor_units, num_outer_units),
                                   name="V1_bias_{}".format(gate_name)))

        ####ingate####
        (self.V1_in_to_ingate,
         self.V1_hid_to_ingate,
         self.V1_bias_ingate) = add_fact2outer_params(outer_ingate, 'ingate')

        ####outgate####
        (self.V1_in_to_forgetgate,
         self.V1_hid_to_forgetgate,
         self.V1_bias_forgetgate) = add_fact2outer_params(outer_forgetgate, 'forgetgate')

        ####cell####
        (self.V1_in_to_cell,
         self.V1_hid_to_cell,
         self.V1_bias_cell) = add_fact2outer_params(outer_cell, 'cell')

        ####outgate####
        (self.V1_in_to_outgate,
         self.V1_hid_to_outgate,
         self.V1_bias_outgate) = add_fact2outer_params(outer_outgate, 'outgate')

        ##############
        # outer loop #
        ##############
        def add_outer_gate_params(gate, gate_name):
            return (self.add_param(gate.W_in,
                                   shape=(num_inputs, num_outer_units),
                                   name="W_outer_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid,
                                   shape=(num_outer_units, num_outer_units),
                                   name="W_outer_hid_to_{}".format(gate_name)),
                    gate.nonlinearity)

        ####ingate####
        (self.W_outer_in_to_ingate,
         self.W_outer_hid_to_ingate,
         self.nonlinearity_outer_ingate) = add_outer_gate_params(outer_ingate, 'ingate')

        ####forgetgate####
        (self.W_outer_in_to_forgetgate,
         self.W_outer_hid_to_forgetgate,
         self.nonlinearity_outer_forgetgate) = add_outer_gate_params(outer_forgetgate, 'forgetgate')

        ####cell####
        (self.W_outer_in_to_cell,
         self.W_outer_hid_to_cell,
         self.nonlinearity_outer_cell) = add_outer_gate_params(outer_cell, 'cell')

        ####outgate####
        (self.W_outer_in_to_outgate,
         self.W_outer_hid_to_outgate,
         self.nonlinearity_outer_outgate) = add_outer_gate_params(outer_outgate, 'outgate')

        ####peephole####
        if self.peepholes:
            self.W_outer_cell_to_ingate = self.add_param(outer_ingate.W_cell,
                                                         shape=(num_outer_units, ),
                                                         name="W_outer_cell_to_ingate")

            self.W_outer_cell_to_forgetgate = self.add_param(outer_forgetgate.W_cell,
                                                             shape=(num_outer_units, ),
                                                             name="W_outer_cell_to_forgetgate")

            self.W_outer_cell_to_outgate = self.add_param(outer_outgate.W_cell,
                                                          shape=(num_outer_units, ),
                                                          name="W_outer_cell_to_outgate")

        ####layer_norm####
        if self.use_layer_norm:
            self.alpha_outer_gate = self.add_param(init.Constant(1.),
                                                   shape=(num_outer_units*4,),
                                                   name="alpha_outer_gate")
            self.beta_outer_gate =  self.add_param(init.Constant(0.),
                                                   shape=(num_outer_units*4,),
                                                   name="beta_outer_gate",
                                                   regularizable=False)

            self.alpha_outer_cell = self.add_param(init.Constant(1.),
                                                   shape=(num_outer_units,),
                                                   name="alpha_outer_cell")
            self.beta_outer_cell =  self.add_param(init.Constant(0.),
                                                   shape=(num_outer_units,),
                                                   name="beta_outer_cell",
                                                   regularizable=False)

        ####init_cell####
        if isinstance(outer_cell_init, Layer):
            self.outer_cell_init = outer_cell_init
        else:
            self.outer_cell_init = self.add_param(outer_cell_init,
                                                  shape=(1, num_outer_units),
                                                  name="outer_cell_init",
                                                  trainable=learn_init,
                                                  regularizable=False)

        ####init_hid####
        if isinstance(outer_hid_init, Layer):
            self.outer_hid_init = outer_hid_init
        else:
            self.outer_hid_init = self.add_param(outer_hid_init,
                                                 shape=(1, num_outer_units),
                                                 name="outer_hid_init",
                                                 trainable=learn_init,
                                                 regularizable=False)

    def layer_norm(self, input, alpha, beta):
        output = (input - T.mean(input, axis=1, keepdims=True))/(T.sqrt(T.var(input, axis=1, keepdims=True)) + eps)
        output = alpha[None, :]*output + beta[None, :]
        return output

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        num_outputs = self.num_outer_units

        if self.only_return_final:
            return input_shape[0], num_outputs
        else:
            return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # input
        input = inputs[0]

        # mask
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # inner hid
        inner_hid_init = None
        if self.inner_hid_init_incoming_index > 0:
            inner_hid_init = inputs[self.inner_hid_init_incoming_index]

        # inner cell
        inner_cell_init = None
        if self.inner_cell_init_incoming_index > 0:
            inner_cell_init = inputs[self.inner_cell_init_incoming_index]

        # outer hid
        outer_hid_init = None
        if self.outer_hid_init_incoming_index > 0:
            outer_hid_init = inputs[self.outer_hid_init_incoming_index]

        # outer cell
        outer_cell_init = None
        if self.outer_cell_init_incoming_index > 0:
            outer_cell_init = inputs[self.outer_cell_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        ##############
        # inner loop #
        ##############

        ####input####
        W_inner_in_stacked = T.concatenate([self.W_inner_in_to_ingate,
                                            self.W_inner_in_to_forgetgate,
                                            self.W_inner_in_to_cell,
                                            self.W_inner_in_to_outgate], axis=1)

        ####hidden####
        W_inner_hid_stacked = T.concatenate([self.W_inner_hid_to_ingate,
                                             self.W_inner_hid_to_forgetgate,
                                             self.W_inner_hid_to_cell,
                                             self.W_inner_hid_to_outgate], axis=1)

        ####bias####
        b_inner_stacked = T.concatenate([self.b_inner_ingate,
                                         self.b_inner_forgetgate,
                                         self.b_inner_cell,
                                         self.b_inner_outgate], axis=0)

        ###################
        # inner to factor #
        ###################

        ####input####
        V0_in_stacked = T.concatenate([self.V0_in_to_ingate,
                                       self.V0_in_to_forgetgate,
                                       self.V0_in_to_cell,
                                       self.V0_in_to_outgate], axis=1)

        b0_in_stacked = T.concatenate([self.b0_in_to_ingate,
                                       self.b0_in_to_forgetgate,
                                       self.b0_in_to_cell,
                                       self.b0_in_to_outgate], axis=0)

        ####hidden####
        V0_hid_stacked = T.concatenate([self.V0_hid_to_ingate,
                                        self.V0_hid_to_forgetgate,
                                        self.V0_hid_to_cell,
                                        self.V0_hid_to_outgate], axis=1)

        b0_hid_stacked = T.concatenate([self.b0_hid_to_ingate,
                                        self.b0_hid_to_forgetgate,
                                        self.b0_hid_to_cell,
                                        self.b0_hid_to_outgate], axis=0)

        ####bias####
        V0_bias_stacked = T.concatenate([self.V0_bias_ingate,
                                         self.V0_bias_forgetgate,
                                         self.V0_bias_cell,
                                         self.V0_bias_outgate], axis=1)

        ##############
        # outer loop #
        ##############

        ####input####
        W_outer_in_stacked = T.concatenate([self.W_outer_in_to_ingate,
                                            self.W_outer_in_to_forgetgate,
                                            self.W_outer_in_to_cell,
                                            self.W_outer_in_to_outgate], axis=1)

        ####hidden####
        W_outer_hid_stacked = T.concatenate([self.W_outer_hid_to_ingate,
                                             self.W_outer_hid_to_forgetgate,
                                             self.W_outer_hid_to_cell,
                                             self.W_outer_hid_to_outgate], axis=1)

        # pre-compute inner/outer input
        if self.precompute_input:
            inner_input = T.dot(input, W_inner_in_stacked) + b_inner_stacked
            outer_input = T.dot(input, W_outer_in_stacked)
        else:
            inner_input = input
            outer_input = input

        # slice for inner
        def slice_inner(x, n):
            return x[:, n*self.num_inner_units:(n+1)*self.num_inner_units]

        # slice for outer
        def slice_outer(x, n):
            return x[:, n*self.num_outer_units:(n+1)*self.num_outer_units]

        # slice for factor
        def slice_factor(x, n):
            return x[:, n*self.num_inner_factor_units:(n+1)*self.num_inner_factor_units]

        # if using dropout
        if deterministic:
            self.using_dropout = False
        else:
            self.using_dropout = True
        inner_cell_mask = self.binomial((num_batch, self.num_inner_units),
                                        p=T.constant(1) - self.p,
                                        dtype=floatX)

        outer_cell_mask = self.binomial((num_batch, self.num_outer_units),
                                        p=T.constant(1) - self.p,
                                        dtype=floatX)

        # step function
        def step(inner_input,
                 outer_input,
                 inner_cell_previous,
                 inner_hid_previous,
                 outer_cell_previous,
                 outer_hid_previous,
                 *args):
            ##############
            # inner loop #
            ##############
            if not self.precompute_input:
                inner_input = T.dot(inner_input, W_inner_in_stacked) + b_inner_stacked
            inner_gates = inner_input
            inner_gates += T.dot(inner_hid_previous, W_inner_hid_stacked)

            if self.use_layer_norm:
                inner_gates = self.layer_norm(input=inner_gates,
                                              alpha=self.alpha_inner_gate,
                                              beta=self.beta_inner_gate)

            if self.grad_clipping:
                inner_gates = theano.gradient.grad_clip(inner_gates,
                                                        -self.grad_clipping,
                                                        self.grad_clipping)

            # get gate slices
            inner_ingate = slice_inner(inner_gates, 0)
            inner_forgetgate = slice_inner(inner_gates, 1)
            inner_cell_input = slice_inner(inner_gates, 2)
            inner_outgate = slice_inner(inner_gates, 3)

            # get peepholes
            if self.peepholes:
                inner_ingate += inner_cell_previous*self.W_inner_cell_to_ingate
                inner_forgetgate += inner_cell_previous*self.W_inner_cell_to_forgetgate

            # get gate nonlinear
            inner_ingate = self.nonlinearity_inner_ingate(inner_ingate)
            inner_forgetgate = self.nonlinearity_inner_forgetgate(inner_forgetgate)
            inner_cell_input = self.nonlinearity_inner_cell(inner_cell_input)

            # drop out
            if self.using_dropout==False or self.p == 0:
                inner_cell_input = inner_cell_input
            else:
                one = T.constant(1)
                retain_prob = one - self.p
                inner_cell_input /= retain_prob
                inner_cell_input = inner_cell_input*inner_cell_mask

            # update inner cell
            inner_cell = inner_forgetgate*inner_cell_previous + inner_ingate*inner_cell_input

            if self.peepholes:
                inner_outgate += inner_cell*self.W_inner_cell_to_outgate
            inner_outgate = self.nonlinearity_inner_outgate(inner_outgate)

            if self.use_layer_norm:
                _cell = self.layer_norm(input=inner_cell,
                                        alpha=self.alpha_inner_cell,
                                        beta=self.beta_inner_cell)
            else:
                _cell = inner_cell

            # update inner hidden
            inner_hid = inner_outgate*self.nonlinearity(_cell)


            ###################
            # inner to factor #
            ###################
            fact_in = T.dot(inner_hid, V0_in_stacked) + b0_in_stacked
            fact_hid = T.dot(inner_hid, V0_hid_stacked) + b0_hid_stacked
            fact_bias = T.dot(inner_hid, V0_bias_stacked)


            ###################
            # factor to outer #
            ###################
            scale_outer_in = T.concatenate([T.dot(slice_factor(fact_in, 0), self.V1_in_to_ingate),
                                            T.dot(slice_factor(fact_in, 1), self.V1_in_to_forgetgate),
                                            T.dot(slice_factor(fact_in, 2), self.V1_in_to_cell),
                                            T.dot(slice_factor(fact_in, 3), self.V1_in_to_outgate)],
                                           axis=1)
            scale_outer_hid = T.concatenate([T.dot(slice_factor(fact_hid, 0), self.V1_hid_to_ingate),
                                             T.dot(slice_factor(fact_hid, 1), self.V1_hid_to_forgetgate),
                                             T.dot(slice_factor(fact_hid, 2), self.V1_hid_to_cell),
                                             T.dot(slice_factor(fact_hid, 3), self.V1_hid_to_outgate)],
                                            axis=1)

            outer_bias = T.concatenate([T.dot(slice_factor(fact_bias, 0), self.V1_bias_ingate),
                                        T.dot(slice_factor(fact_bias, 1), self.V1_bias_forgetgate),
                                        T.dot(slice_factor(fact_bias, 2), self.V1_bias_cell),
                                        T.dot(slice_factor(fact_bias, 3), self.V1_bias_outgate)],
                                       axis=1)

            ##############
            # outer loop #
            ##############
            if not self.precompute_input:
                outer_input = T.dot(outer_input, W_outer_in_stacked)
            outer_gates = scale_outer_in*outer_input
            outer_gates += scale_outer_hid*T.dot(outer_hid_previous, W_outer_hid_stacked)
            outer_gates += outer_bias

            if self.use_layer_norm:
                outer_gates = self.layer_norm(input=outer_gates,
                                              alpha=self.alpha_outer_gate,
                                              beta=self.beta_outer_gate)

            if self.grad_clipping:
                outer_gates = theano.gradient.grad_clip(outer_gates,
                                                        -self.grad_clipping,
                                                        self.grad_clipping)

            # get gate slices
            outer_ingate = slice_outer(outer_gates, 0)
            outer_forgetgate = slice_outer(outer_gates, 1)
            outer_cell_input = slice_outer(outer_gates, 2)
            outer_outgate = slice_outer(outer_gates, 3)

            # get peepholes
            if self.peepholes:
                outer_ingate += outer_cell_previous*self.W_outer_cell_to_ingate
                outer_forgetgate += outer_cell_previous*self.W_outer_cell_to_forgetgate

            # get gate nonlinear
            outer_ingate = self.nonlinearity_outer_ingate(outer_ingate)
            outer_forgetgate = self.nonlinearity_outer_forgetgate(outer_forgetgate)
            outer_cell_input = self.nonlinearity_outer_cell(outer_cell_input)

            # drop out
            if self.using_dropout==False or self.p == 0:
                outer_cell_input = outer_cell_input
            else:
                one = T.constant(1)
                retain_prob = one - self.p
                outer_cell_input /= retain_prob
                outer_cell_input = outer_cell_input*outer_cell_mask

            # update inner cell
            outer_cell = outer_forgetgate*outer_cell_previous + outer_ingate*outer_cell_input

            if self.peepholes:
                outer_outgate += outer_cell*self.W_outer_cell_to_outgate
            outer_outgate = self.nonlinearity_outer_outgate(outer_outgate)

            if self.use_layer_norm:
                _cell = self.layer_norm(input=outer_cell,
                                        alpha=self.alpha_outer_cell,
                                        beta=self.beta_outer_cell)
            else:
                _cell = outer_cell

            outer_hid = outer_outgate*self.nonlinearity(_cell)
            return [inner_cell, inner_hid, outer_cell, outer_hid]

        def step_masked(inner_input,
                        outer_input,
                        mask_n,
                        inner_cell_previous,
                        inner_hid_previous,
                        outer_cell_previous,
                        outer_hid_previous,
                        *args):
            inner_cell, inner_hid, outer_cell, outer_hid = step(inner_input,
                                                                outer_input,
                                                                inner_cell_previous,
                                                                inner_hid_previous,
                                                                outer_cell_previous,
                                                                outer_hid_previous,
                                                                *args)

            inner_cell = T.switch(mask_n, inner_cell, inner_cell_previous)
            inner_hid = T.switch(mask_n, inner_hid, inner_hid_previous)
            outer_cell = T.switch(mask_n, outer_cell, outer_cell_previous)
            outer_hid = T.switch(mask_n, outer_hid, outer_hid_previous)
            return [inner_cell, inner_hid, outer_cell, outer_hid]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [inner_input, outer_input, mask]
            step_fun = step_masked
        else:
            sequences = [inner_input, outer_input]
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.inner_cell_init, Layer):
            inner_cell_init = T.dot(ones, self.inner_cell_init)

        if not isinstance(self.inner_hid_init, Layer):
            inner_hid_init = T.dot(ones, self.inner_hid_init)

        if not isinstance(self.outer_cell_init, Layer):
            outer_cell_init = T.dot(ones, self.outer_cell_init)

        if not isinstance(self.outer_hid_init, Layer):
            outer_hid_init = T.dot(ones, self.outer_hid_init)

        non_seqs = [inner_cell_mask, outer_cell_mask]
        if not self.precompute_input:
            non_seqs += [W_inner_in_stacked, b_inner_stacked,
                         W_outer_in_stacked]

        non_seqs += [W_inner_hid_stacked, W_outer_hid_stacked,
                     V0_in_stacked, b0_in_stacked,
                     V0_hid_stacked, b0_hid_stacked,
                     V0_bias_stacked]

        non_seqs += [self.V1_in_to_ingate,
                     self.V1_in_to_forgetgate,
                     self.V1_in_to_cell,
                     self.V1_in_to_outgate,
                     self.V1_hid_to_ingate,
                     self.V1_hid_to_forgetgate,
                     self.V1_hid_to_cell,
                     self.V1_hid_to_outgate,
                     self.V1_bias_ingate,
                     self.V1_bias_forgetgate,
                     self.V1_bias_cell,
                     self.V1_bias_outgate]

        if self.peepholes:
            non_seqs += [self.W_inner_cell_to_ingate,
                         self.W_inner_cell_to_forgetgate,
                         self.W_inner_cell_to_outgate,
                         self.W_outer_cell_to_ingate,
                         self.W_outer_cell_to_forgetgate,
                         self.W_outer_cell_to_outgate]

        if self.use_layer_norm:
            non_seqs +=[self.alpha_inner_gate,
                        self.alpha_inner_cell,
                        self.beta_inner_gate,
                        self.beta_inner_cell]

            non_seqs +=[self.alpha_outer_gate,
                        self.alpha_outer_cell,
                        self.beta_outer_gate,
                        self.beta_outer_cell]
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            inner_cell_out, inner_hid_out, outer_cell_out, outer_hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[inner_cell_init, inner_hid_init, outer_cell_init, outer_hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            inner_cell_out, inner_hid_out, outer_cell_out, outer_hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[inner_cell_init, inner_hid_init, outer_cell_init, outer_hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = outer_hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = outer_hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = outer_hid_out[:, ::-1]

        return hid_out


class BiDirScaleHyperLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_inner_units,
                 num_inner_factor_units,
                 num_outer_units,
                 inner_ingate=Gate(),
                 inner_forgetgate=Gate(b=init.Constant(1.)),
                 inner_cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 inner_outgate=Gate(),
                 outer_ingate=Gate(),
                 outer_forgetgate=Gate(b=init.Constant(1.)),
                 outer_cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outer_outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 fwd_inner_cell_init=init.Constant(0.),
                 fwd_inner_hid_init=init.Constant(0.),
                 fwd_outer_cell_init=init.Constant(0.),
                 fwd_outer_hid_init=init.Constant(0.),
                 bwd_inner_cell_init=init.Constant(0.),
                 bwd_inner_hid_init=init.Constant(0.),
                 bwd_outer_cell_init=init.Constant(0.),
                 bwd_outer_hid_init=init.Constant(0.),
                 dropout_ratio=0.2,
                 use_layer_norm=True,
                 weight_noise=0.0,
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        incomings = [incoming]

        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-11

        self.fwd_inner_hid_init_incoming_index = -1
        if isinstance(fwd_inner_hid_init, Layer):
            incomings.append(fwd_inner_hid_init)
            self.fwd_inner_hid_init_incoming_index = len(incomings)-1

        self.fwd_inner_cell_init_incoming_index = -1
        if isinstance(fwd_inner_cell_init, Layer):
            incomings.append(fwd_inner_cell_init)
            self.fwd_inner_cell_init_incoming_index = len(incomings)-1

        self.fwd_outer_hid_init_incoming_index = -1
        if isinstance(fwd_outer_hid_init, Layer):
            incomings.append(fwd_outer_hid_init)
            self.fwd_outer_hid_init_incoming_index = len(incomings)-1

        self.fwd_outer_cell_init_incoming_index = -1
        if isinstance(fwd_outer_cell_init, Layer):
            incomings.append(fwd_outer_cell_init)
            self.fwd_outer_cell_init_incoming_index = len(incomings)-1

        self.bwd_inner_hid_init_incoming_index = -1
        if isinstance(bwd_inner_hid_init, Layer):
            incomings.append(bwd_inner_hid_init)
            self.bwd_inner_hid_init_incoming_index = len(incomings)-1

        self.bwd_inner_cell_init_incoming_index = -1
        if isinstance(bwd_inner_cell_init, Layer):
            incomings.append(bwd_inner_cell_init)
            self.bwd_inner_cell_init_incoming_index = len(incomings)-1

        self.bwd_outer_hid_init_incoming_index = -1
        if isinstance(bwd_outer_hid_init, Layer):
            incomings.append(bwd_outer_hid_init)
            self.bwd_outer_hid_init_incoming_index = len(incomings)-1

        self.bwd_outer_cell_init_incoming_index = -1
        if isinstance(bwd_outer_cell_init, Layer):
            incomings.append(bwd_outer_cell_init)
            self.bwd_outer_cell_init_incoming_index = len(incomings)-1

        super(BiDirScaleHyperLSTMLayer, self).__init__(incomings, **kwargs)

        self.fwd_lstm_layer = ScaleHyperLSTMLayer(incoming=incoming,
                                                  num_inner_units=num_inner_units,
                                                  num_inner_factor_units=num_inner_factor_units,
                                                  num_outer_units=num_outer_units,
                                                  inner_ingate=inner_ingate,
                                                  inner_forgetgate=inner_forgetgate,
                                                  inner_cell=inner_cell,
                                                  inner_outgate=inner_outgate,
                                                  outer_ingate=outer_ingate,
                                                  outer_forgetgate=outer_forgetgate,
                                                  outer_cell=outer_cell,
                                                  outer_outgate=outer_outgate,
                                                  nonlinearity=nonlinearity,
                                                  inner_cell_init=fwd_inner_cell_init,
                                                  inner_hid_init=fwd_inner_hid_init,
                                                  outer_cell_init=fwd_outer_cell_init,
                                                  outer_hid_init=fwd_outer_hid_init,
                                                  dropout_ratio=dropout_ratio,
                                                  use_layer_norm=use_layer_norm,
                                                  weight_noise=weight_noise,
                                                  backwards=False,
                                                  learn_init=learn_init,
                                                  peepholes=peepholes,
                                                  gradient_steps=gradient_steps,
                                                  grad_clipping=grad_clipping,
                                                  unroll_scan=unroll_scan,
                                                  precompute_input=precompute_input,
                                                  mask_input=mask_input,
                                                  only_return_final=only_return_final,
                                                  **kwargs)
        self.params.update(self.fwd_lstm_layer.params)

        self.bwd_lstm_layer = ScaleHyperLSTMLayer(incoming=incoming,
                                                  num_inner_units=num_inner_units,
                                                  num_inner_factor_units=num_inner_factor_units,
                                                  num_outer_units=num_outer_units,
                                                  inner_ingate=inner_ingate,
                                                  inner_forgetgate=inner_forgetgate,
                                                  inner_cell=inner_cell,
                                                  inner_outgate=inner_outgate,
                                                  outer_ingate=outer_ingate,
                                                  outer_forgetgate=outer_forgetgate,
                                                  outer_cell=outer_cell,
                                                  outer_outgate=outer_outgate,
                                                  nonlinearity=nonlinearity,
                                                  inner_cell_init=fwd_inner_cell_init,
                                                  inner_hid_init=fwd_inner_hid_init,
                                                  outer_cell_init=fwd_outer_cell_init,
                                                  outer_hid_init=fwd_outer_hid_init,
                                                  dropout_ratio=dropout_ratio,
                                                  use_layer_norm=use_layer_norm,
                                                  weight_noise=weight_noise,
                                                  backwards=True,
                                                  learn_init=learn_init,
                                                  peepholes=peepholes,
                                                  gradient_steps=gradient_steps,
                                                  grad_clipping=grad_clipping,
                                                  unroll_scan=unroll_scan,
                                                  precompute_input=precompute_input,
                                                  mask_input=mask_input,
                                                  only_return_final=only_return_final,
                                                  **kwargs)
        self.params.update(self.bwd_lstm_layer.params)

        self.output_layer = ConcatLayer(incomings=[self.fwd_lstm_layer,
                                                   self.bwd_lstm_layer],
                                        axis=-1)

        self.num_units = num_outer_units*2
        self.only_return_final = only_return_final

    def get_output_shape_for(self, input_shapes):
        num_samples = input_shapes[0][0]
        num_step = input_shapes[0][1]
        num_units = self.num_units

        if self.only_return_final:
            return (num_samples, num_units)
        else:
            return (num_samples, num_step, num_units)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input_dict = {}

        input = inputs[0]
        input_dict[self.input_layers[0]] = input

        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
            input_dict[self.input_layers[self.mask_incoming_index]] = mask

        if self.fwd_inner_hid_init_incoming_index > 0:
            fwd_inner_hid_init = inputs[self.fwd_inner_hid_init_incoming_index]
            input_dict[self.input_layers[self.fwd_inner_hid_init_incoming_index]] = fwd_inner_hid_init

        if self.fwd_inner_cell_init_incoming_index > 0:
            fwd_inner_cell_init = inputs[self.fwd_inner_cell_init_incoming_index]
            input_dict[self.input_layers[self.fwd_inner_cell_init_incoming_index]] = fwd_inner_cell_init

        if self.fwd_outer_hid_init_incoming_index > 0:
            fwd_outer_hid_init = inputs[self.fwd_outer_hid_init_incoming_index]
            input_dict[self.input_layers[self.fwd_outer_hid_init_incoming_index]] = fwd_outer_hid_init

        if self.fwd_outer_cell_init_incoming_index > 0:
            fwd_outer_cell_init = inputs[self.fwd_outer_cell_init_incoming_index]
            input_dict[self.input_layers[self.fwd_outer_cell_init_incoming_index]] = fwd_outer_cell_init

        if self.bwd_inner_hid_init_incoming_index > 0:
            bwd_inner_hid_init = inputs[self.bwd_inner_hid_init_incoming_index]
            input_dict[self.input_layers[self.bwd_inner_hid_init_incoming_index]] = bwd_inner_hid_init

        if self.bwd_inner_cell_init_incoming_index > 0:
            bwd_inner_cell_init = inputs[self.bwd_inner_cell_init_incoming_index]
            input_dict[self.input_layers[self.bwd_inner_cell_init_incoming_index]] = bwd_inner_cell_init

        if self.bwd_outer_hid_init_incoming_index > 0:
            bwd_outer_hid_init = inputs[self.bwd_outer_hid_init_incoming_index]
            input_dict[self.input_layers[self.bwd_outer_hid_init_incoming_index]] = bwd_outer_hid_init

        if self.bwd_outer_cell_init_incoming_index > 0:
            bwd_outer_cell_init = inputs[self.bwd_outer_cell_init_incoming_index]
            input_dict[self.input_layers[self.bwd_outer_cell_init_incoming_index]] = bwd_outer_cell_init


        output = get_output(self.output_layer,
                            inputs=input_dict,
                            deterministic=deterministic)

        return output