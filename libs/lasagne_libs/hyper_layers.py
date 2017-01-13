import theano
import numpy
from theano import tensor as T
from lasagne import init, nonlinearities
from lasagne.layers import Layer, MergeLayer, Gate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from lasagne.utils import unroll_scan
floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

class ScalingHyperLSTMLayer(MergeLayer):
    def __init__(self,
                 inner_incoming,
                 outer_incoming,
                 mask_incoming,
                 num_inner_units,
                 num_outer_units,
                 use_peepholes=False,
                 use_layer_norm=False,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=1,
                 only_return_outer=False,
                 **kwargs):

        # set input layers
        incomings = [inner_incoming,
                     outer_incoming,
                     mask_incoming]
        # initialize
        super(ScalingHyperLSTMLayer, self).__init__(incomings, **kwargs)

        # set options
        self.num_inner_units = num_inner_units
        self.num_outer_units = num_outer_units
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_outer = only_return_outer
        self.use_peepholes = use_peepholes

        # for layer norm
        self.use_layer_norm = use_layer_norm

        # get size
        inner_input_shape = self.input_shapes[0]
        outer_input_shape = self.input_shapes[1]
        num_inner_inputs = inner_input_shape[2]
        num_outer_inputs = outer_input_shape[2]

        ##############
        # inner loop #
        ##############
        def add_inner_gate_params(gate_name,
                                  cell_trainable=True,
                                  use_layer_norm=False,
                                  bias_const=0.0):
            return (#### inner input-to-inner ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inner_inputs, num_inner_units),
                                   name="W_inner_in_to_inner_{}".format(gate_name)),

                    #### inner hidden-to-inner ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inner_units, num_inner_units),
                                   name="W_inner_hid_to_inner_{}".format(gate_name)),

                    #### inner cell-to-inner ####
                    self.add_param(spec=init.Uniform(0.1) if cell_trainable else init.Constant(0.0),
                                   shape=(num_inner_units,),
                                   name="W_inner_cell_to_inner_{}".format(gate_name),
                                   trainable=cell_trainable),

                    #### outer hidden-to-inner ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inner_units, num_inner_units),
                                   name="W_outer_hid_to_inner_{}".format(gate_name)),

                    #### bias ####
                    self.add_param(spec=init.Constant(bias_const),
                                   shape=(num_inner_units,),
                                   name="b_inner_{}".format(gate_name),
                                   regularizable=False),

                    #### layer norm ####
                    self.add_param(spec=init.Constant(1.),
                                   shape=(num_inner_units,),
                                   name="W_inner_ln_{}".format(gate_name),
                                   trainable=use_layer_norm),
                    self.add_param(spec=init.Constant(0.),
                                   shape=(num_inner_units,),
                                   name="b_inner_ln_{}".format(gate_name),
                                   trainable=use_layer_norm,
                                   regularizable=False))

        ####ingate####
        (self.W_inner_in_to_inner_ingate,
         self.W_inner_hid_to_inner_ingate,
         self.W_inner_cell_to_inner_ingate,
         self.W_outer_hid_to_inner_ingate,
         self.b_inner_ingate,
         self.W_inner_ln_ingate,
         self.b_inner_ln_ingate) = add_inner_gate_params(gate_name='ingate',
                                                         cell_trainable=use_peepholes,
                                                         use_layer_norm=use_layer_norm)

        ####forgetgate#####
        (self.W_inner_in_to_inner_forgetgate,
         self.W_inner_hid_to_inner_forgetgate,
         self.W_inner_cell_to_inner_forgetgate,
         self.W_outer_hid_to_inner_forgetgate,
         self.b_inner_forgetgate,
         self.W_inner_ln_forgetgate,
         self.b_inner_ln_forgetgate) = add_inner_gate_params(gate_name='forgetgate',
                                                             cell_trainable=use_peepholes,
                                                             use_layer_norm=use_layer_norm,
                                                             bias_const=1.0)

        ####cell#####
        (self.W_inner_in_to_inner_cell,
         self.W_inner_hid_to_inner_cell,
         self.W_inner_cell_to_inner_cell,
         self.W_outer_hid_to_inner_cell,
         self.b_inner_cell,
         self.W_inner_ln_cell,
         self.b_inner_ln_cell) = add_inner_gate_params(gate_name='cell',
                                                       cell_trainable=False,
                                                       use_layer_norm=use_layer_norm)

        ####outgate#####
        (self.W_inner_in_to_inner_outgate,
         self.W_inner_hid_to_inner_outgate,
         self.W_inner_cell_to_inner_outgate,
         self.W_outer_hid_to_inner_outgate,
         self.b_inner_outgate,
         self.W_inner_ln_outgate,
         self.b_inner_ln_outgate) = add_inner_gate_params(gate_name='outgate',
                                                          cell_trainable=use_peepholes,
                                                          use_layer_norm=use_layer_norm)

        ####out cell#####
        self.W_inner_ln_outcell = self.add_param(init.Constant(1.),
                                                 shape=(num_inner_units,),
                                                 name="W_inner_ln_outcell",
                                                 trainable=use_layer_norm)
        self.b_inner_ln_outcell =  self.add_param(init.Constant(0.),
                                                  shape=(num_inner_units,),
                                                  name="b_inner_ln_outcell",
                                                  trainable=use_layer_norm,
                                                  regularizable=False)

        ####init_cell####
        self.inner_cell_init = self.add_param(init.Constant(0.),
                                              shape=(1, num_inner_units),
                                              name="inner_cell_init",
                                              trainable=learn_init,
                                              regularizable=False)

        ####init_hidden####
        self.inner_hid_init = self.add_param(init.Constant(0.),
                                             shape=(1, num_inner_units),
                                             name="inner_hid_init",
                                             trainable=learn_init,
                                             regularizable=False)

        ##############
        # outer loop #
        ##############
        def add_outer_gate_params(gate_name,
                                  cell_trainable=True,
                                  use_layer_norm=False,
                                  bias_const=0.0):
            return (#### outer input-to-hidden ####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_outer_inputs, num_outer_units),
                                   name="W_outer_in_to_outer_{}".format(gate_name)),
                    #### inner hidden-to-outer in scale ####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_inner_hid_to_outer_in_{}".format(gate_name)),
                    self.add_param(init.Constant(1.0),
                                   shape=(num_outer_units,),
                                   name="W_inner_hid_to_outer_in_{}".format(gate_name),
                                   regularizable=False),

                    #### outer hidden-to-hidden ####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_outer_hid_to_outer_{}".format(gate_name)),
                    #### inner hidden-to-outer hidden scale ####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_inner_hid_to_outer_hid_{}".format(gate_name)),
                    self.add_param(init.Constant(1.0),
                                   shape=(num_outer_units,),
                                   name="W_inner_hid_to_outer_hid_{}".format(gate_name),
                                   regularizable=False),

                    #### inner hidden-to-outer cell scale ####
                    self.add_param(init.Orthogonal() if cell_trainable else init.Constant(0.0),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_inner_hid_to_outer_cell_{}".format(gate_name),
                                   trainable=cell_trainable),
                    self.add_param(init.Constant(1.0) if cell_trainable else init.Constant(0.0),
                                   shape=(num_outer_units,),
                                   name="W_inner_hid_to_outer_cell_{}".format(gate_name),
                                   trainable=cell_trainable,
                                   regularizable=False),

                    #### inner hidden-to-outer bias ####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_inner_hid_to_outer_bias_{}".format(gate_name)),
                    self.add_param(init.Constant(bias_const),
                                   shape=(num_outer_units,),
                                   name="W_inner_hid_to_outer_bias_{}".format(gate_name),
                                   regularizable=False),

                    #### layer norm ####
                    self.add_param(init.Constant(1.),
                                   shape=(num_outer_units,),
                                   name="W_outer_ln_{}".format(gate_name),
                                   trainable=use_layer_norm),
                    self.add_param(init.Constant(0.),
                                   shape=(num_outer_units,),
                                   name="b_outer_ln_{}".format(gate_name),
                                   trainable=use_layer_norm,
                                   regularizable=False))

        ####ingate####
        (self.W_outer_in_to_outer_ingate,
         self.W_inner_hid_to_outer_in_ingate,
         self.b_inner_hid_to_outer_in_ingate,
         self.W_outer_hid_to_outer_ingate,
         self.W_inner_hid_to_outer_hid_ingate,
         self.b_inner_hid_to_outer_hid_ingate,
         self.W_inner_hid_to_outer_cell_ingate,
         self.b_inner_hid_to_outer_cell_ingate,
         self.W_inner_hid_to_outer_bias_ingate,
         self.b_inner_hid_to_outer_bias_ingate,
         self.W_outer_ln_ingate,
         self.b_outer_ln_ingate) = add_outer_gate_params(gate_name='ingate',
                                                         cell_trainable=use_peepholes,
                                                         use_layer_norm=use_layer_norm)

        ####forgetgate####
        (self.W_outer_in_to_outer_forgetgate,
         self.W_inner_hid_to_outer_in_forgetgate,
         self.b_inner_hid_to_outer_in_forgetgate,
         self.W_outer_hid_to_outer_forgetgate,
         self.W_inner_hid_to_outer_hid_forgetgate,
         self.b_inner_hid_to_outer_hid_forgetgate,
         self.W_inner_hid_to_outer_cell_forgetgate,
         self.b_inner_hid_to_outer_cell_forgetgate,
         self.W_inner_hid_to_outer_bias_forgetgate,
         self.b_inner_hid_to_outer_bias_forgetgate,
         self.W_outer_ln_forgetgate,
         self.b_outer_ln_forgetgate) = add_outer_gate_params(gate_name='forgetgate',
                                                             cell_trainable=use_peepholes,
                                                             use_layer_norm=use_layer_norm,
                                                             bias_const=1.0)

        ####cell####
        (self.W_outer_in_to_outer_cell,
         self.W_inner_hid_to_outer_in_cell,
         self.b_inner_hid_to_outer_in_cell,
         self.W_outer_hid_to_outer_cell,
         self.W_inner_hid_to_outer_hid_cell,
         self.b_inner_hid_to_outer_hid_cell,
         self.W_inner_hid_to_outer_cell_cell,
         self.b_inner_hid_to_outer_cell_cell,
         self.W_inner_hid_to_outer_bias_cell,
         self.b_inner_hid_to_outer_bias_cell,
         self.W_outer_ln_cell,
         self.b_outer_ln_cell) = add_outer_gate_params(gate_name='cell',
                                                       cell_trainable=False,
                                                       use_layer_norm=use_layer_norm,)

        ####outgate####
        (self.W_outer_in_to_outer_outgate,
         self.W_inner_hid_to_outer_in_outgate,
         self.b_inner_hid_to_outer_in_outgate,
         self.W_outer_hid_to_outer_outgate,
         self.W_inner_hid_to_outer_hid_outgate,
         self.b_inner_hid_to_outer_hid_outgate,
         self.W_inner_hid_to_outer_cell_outgate,
         self.b_inner_hid_to_outer_cell_outgate,
         self.W_inner_hid_to_outer_bias_outgate,
         self.b_inner_hid_to_outer_bias_outgate,
         self.W_outer_ln_outgate,
         self.b_outer_ln_outgate) = add_outer_gate_params(gate_name='outgate',
                                                          cell_trainable=use_peepholes,
                                                          use_layer_norm=use_layer_norm)

        ####out cell#####
        self.W_outer_ln_outcell = self.add_param(init.Constant(1.),
                                                 shape=(num_outer_units,),
                                                 name="W_outer_ln_outcell",
                                                 trainable=use_layer_norm)
        self.b_outer_ln_outcell =  self.add_param(init.Constant(0.),
                                                  shape=(num_outer_units,),
                                                  name="b_outer_ln_outcell",
                                                  trainable=use_layer_norm,
                                                  regularizable=False)

        ####hidden project#####
        self.W_outer_hid_prj = self.add_param(init.Orthogonal(0.1),
                                              shape=(num_outer_units, num_inner_units),
                                              name="W_outer_hid_prj")

        ####init_cell####
        self.outer_cell_init = self.add_param(init.Constant(0.),
                                              shape=(1, num_outer_units),
                                              name="outer_cell_init",
                                              trainable=learn_init,
                                              regularizable=False)

        ####init_hid####
        self.outer_hid_init = self.add_param(init.Constant(0.),
                                             shape=(1, num_inner_units),
                                             name="outer_hid_init",
                                             trainable=learn_init,
                                             regularizable=False)

    def layer_norm(self, input, alpha, beta):
        output = (input - T.mean(input, axis=1, keepdims=True))/(T.sqrt(T.var(input, axis=1, keepdims=True)) + eps)
        output = alpha[None, :]*output + beta[None, :]
        return output

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_outer:
            num_outputs = self.num_inner_units
        else:
            num_outputs = self.num_inner_units*2

        return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # input
        inner_input = inputs[0]
        outer_input = inputs[1]
        mask_input = inputs[2]

        inner_input = inner_input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = inner_input.shape
        outer_input = outer_input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = outer_input.shape
        mask_input = mask_input.dimshuffle(1, 0, 'x')

        ##############
        # inner loop #
        ##############
        #### input-to-hidden ####
        W_inner_in_to_inner_concat = T.concatenate([self.W_inner_in_to_inner_ingate,
                                                    self.W_inner_in_to_inner_forgetgate,
                                                    self.W_inner_in_to_inner_cell,
                                                    self.W_inner_in_to_inner_outgate], axis=1)

        #### hidden-to-hidden ####
        W_inner_hid_to_inner_concat = T.concatenate([self.W_inner_hid_to_inner_ingate,
                                                     self.W_inner_hid_to_inner_forgetgate,
                                                     self.W_inner_hid_to_inner_cell,
                                                     self.W_inner_hid_to_inner_outgate], axis=1)

        #### cell-to-hidden ####
        W_inner_cell_to_inner_concat = T.concatenate([self.W_inner_cell_to_inner_ingate,
                                                      self.W_inner_cell_to_inner_forgetgate,
                                                      self.W_inner_cell_to_inner_cell,
                                                      self.W_inner_cell_to_inner_outgate], axis=0)

        #### hidden-to-hidden ####
        W_outer_hid_to_inner_concat = T.concatenate([self.W_outer_hid_to_inner_ingate,
                                                     self.W_outer_hid_to_inner_forgetgate,
                                                     self.W_outer_hid_to_inner_cell,
                                                     self.W_outer_hid_to_inner_outgate], axis=1)

        #### bias ####
        b_inner_concat = T.concatenate([self.b_inner_ingate,
                                        self.b_inner_forgetgate,
                                        self.b_inner_cell,
                                        self.b_inner_outgate], axis=0)

        ##############
        # outer loop #
        ##############
        # inner hid-to-outer in scale
        W_inner_hid_to_outer_in_concat = T.concatenate([self.W_inner_hid_to_outer_in_ingate,
                                                        self.W_inner_hid_to_outer_in_forgetgate,
                                                        self.W_inner_hid_to_outer_in_cell,
                                                        self.W_inner_hid_to_outer_in_outgate], axis=1)
        b_inner_hid_to_outer_in_concat = T.concatenate([self.b_inner_hid_to_outer_in_ingate,
                                                        self.b_inner_hid_to_outer_in_forgetgate,
                                                        self.b_inner_hid_to_outer_in_cell,
                                                        self.b_inner_hid_to_outer_in_outgate], axis=0)

        # inner hid-to-outer hid scale
        W_inner_hid_to_outer_hid_concat = T.concatenate([self.W_inner_hid_to_outer_hid_ingate,
                                                         self.W_inner_hid_to_outer_hid_forgetgate,
                                                         self.W_inner_hid_to_outer_hid_cell,
                                                         self.W_inner_hid_to_outer_hid_outgate], axis=1)
        b_inner_hid_to_outer_hid_concat = T.concatenate([self.b_inner_hid_to_outer_hid_ingate,
                                                         self.b_inner_hid_to_outer_hid_forgetgate,
                                                         self.b_inner_hid_to_outer_hid_cell,
                                                         self.b_inner_hid_to_outer_hid_outgate], axis=0)

        # inner hid-to-outer cell scale
        W_inner_hid_to_outer_cell_concat = T.concatenate([self.W_inner_hid_to_outer_cell_ingate,
                                                          self.W_inner_hid_to_outer_cell_forgetgate,
                                                          self.W_inner_hid_to_outer_cell_cell,
                                                          self.W_inner_hid_to_outer_cell_outgate], axis=1)
        b_inner_hid_to_outer_cell_concat = T.concatenate([self.b_inner_hid_to_outer_cell_ingate,
                                                          self.b_inner_hid_to_outer_cell_forgetgate,
                                                          self.b_inner_hid_to_outer_cell_cell,
                                                          self.b_inner_hid_to_outer_cell_outgate], axis=0)

        # inner hid-to-outer bias
        W_inner_hid_to_outer_bias_concat = T.concatenate([self.W_inner_hid_to_outer_bias_ingate,
                                                          self.W_inner_hid_to_outer_bias_forgetgate,
                                                          self.W_inner_hid_to_outer_bias_cell,
                                                          self.W_inner_hid_to_outer_bias_outgate], axis=1)
        b_inner_hid_to_outer_bias_concat = T.concatenate([self.b_inner_hid_to_outer_bias_ingate,
                                                          self.b_inner_hid_to_outer_bias_forgetgate,
                                                          self.b_inner_hid_to_outer_bias_cell,
                                                          self.b_inner_hid_to_outer_bias_outgate], axis=0)

        #### input-to-hidden ####
        W_outer_in_to_outer_concat = T.concatenate([self.W_outer_in_to_outer_ingate,
                                                    self.W_outer_in_to_outer_forgetgate,
                                                    self.W_outer_in_to_outer_cell,
                                                    self.W_outer_in_to_outer_outgate], axis=1)

        #### hidden-to-hidden ####
        W_outer_hid_to_outer_concat = T.concatenate([self.W_outer_hid_to_outer_ingate,
                                                     self.W_outer_hid_to_outer_forgetgate,
                                                     self.W_outer_hid_to_outer_cell,
                                                     self.W_outer_hid_to_outer_outgate], axis=1)

        # pre-compute inner/outer input
        inner_input = T.dot(inner_input, W_inner_in_to_inner_concat) + b_inner_concat
        outer_input = T.dot(outer_input, W_outer_in_to_outer_concat)

        # slice for inner
        def slice_inner(x, n):
            return x[:, n*self.num_inner_units:(n+1)*self.num_inner_units]

        # slice for outer
        def slice_outer(x, n):
            return x[:, n*self.num_outer_units:(n+1)*self.num_outer_units]

        # step function
        def step(inner_input_n,
                 outer_input_n,
                 inner_cell_previous,
                 inner_hid_previous,
                 outer_cell_previous,
                 outer_hid_previous,
                 *args):
            ##############
            # inner loop #
            ##############
            inner_gates = inner_input_n
            inner_gates += T.dot(inner_hid_previous, W_inner_hid_to_inner_concat)
            inner_gates += T.dot(outer_hid_previous, W_outer_hid_to_inner_concat)

            inner_ingate = slice_inner(inner_gates, 0) + inner_cell_previous*self.W_inner_cell_to_inner_ingate
            inner_forgetgate = slice_inner(inner_gates, 1) + inner_cell_previous*self.W_inner_cell_to_inner_forgetgate
            inner_cell_input = slice_inner(inner_gates, 2) + inner_cell_previous*self.W_inner_cell_to_inner_cell

            if self.use_layer_norm:
                inner_ingate = self.layer_norm(input=inner_ingate,
                                               alpha=self.W_inner_ln_ingate,
                                               beta=self.b_inner_ln_ingate)
                inner_forgetgate = self.layer_norm(input=inner_forgetgate,
                                                   alpha=self.W_inner_ln_forgetgate,
                                                   beta=self.b_inner_ln_forgetgate)
                inner_cell_input = self.layer_norm(input=inner_cell_input,
                                                   alpha=self.W_inner_ln_cell,
                                                   beta=self.b_inner_ln_cell)

            if self.grad_clipping:
                inner_ingate = theano.gradient.grad_clip(inner_ingate,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                inner_forgetgate = theano.gradient.grad_clip(inner_forgetgate,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
                inner_cell_input = theano.gradient.grad_clip(inner_cell_input,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)

            inner_ingate = T.nnet.sigmoid(inner_ingate)
            inner_forgetgate = T.nnet.sigmoid(inner_forgetgate)
            inner_cell_input = T.tanh(inner_cell_input)

            inner_cell = inner_forgetgate*inner_cell_previous + inner_ingate*inner_cell_input

            inner_outgate = slice_inner(inner_gates, 3) + inner_cell*self.W_inner_cell_to_inner_outgate
            if self.use_layer_norm:
                inner_outgate = self.layer_norm(input=inner_outgate,
                                                alpha=self.W_inner_ln_outgate,
                                                beta=self.b_inner_ln_outgate)

            if self.grad_clipping:
                inner_outgate = theano.gradient.grad_clip(inner_outgate,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)
            inner_outgate = T.nnet.sigmoid(inner_outgate)

            inner_outcell = inner_cell
            if self.use_layer_norm:
                inner_outcell = self.layer_norm(input=inner_outcell,
                                                alpha=self.W_inner_ln_outcell,
                                                beta=self.b_inner_ln_outcell)
            if self.grad_clipping:
                inner_outcell = theano.gradient.grad_clip(inner_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            # update inner hidden
            inner_hid = inner_outgate*T.tanh(inner_outcell)

            ###################
            # factor to outer #
            ###################
            scale_outer_in = T.dot(inner_hid, W_inner_hid_to_outer_in_concat) + b_inner_hid_to_outer_in_concat
            scale_outer_hid = T.dot(inner_hid, W_inner_hid_to_outer_hid_concat) + b_inner_hid_to_outer_hid_concat
            scale_outer_cell = T.dot(inner_hid, W_inner_hid_to_outer_cell_concat) + b_inner_hid_to_outer_cell_concat
            scale_outer_bias = T.dot(inner_hid, W_inner_hid_to_outer_bias_concat) + b_inner_hid_to_outer_bias_concat

            ##############
            # outer loop #
            ##############
            outer_gates = scale_outer_in*outer_input_n
            outer_gates += scale_outer_hid*T.dot(outer_hid_previous, W_outer_hid_to_outer_concat)
            outer_gates += scale_outer_bias

            outer_ingate = slice_outer(outer_gates, 0) + outer_cell_previous*slice_outer(scale_outer_cell, 0)
            outer_forgetgate = slice_outer(outer_gates, 1) + outer_cell_previous*slice_outer(scale_outer_cell, 1)
            outer_cell_input = slice_outer(outer_gates, 2) + outer_cell_previous*slice_outer(scale_outer_cell, 2)

            if self.use_layer_norm:
                outer_ingate = self.layer_norm(input=outer_ingate,
                                               alpha=self.W_outer_ln_ingate,
                                               beta=self.b_outer_ln_ingate)
                outer_forgetgate = self.layer_norm(input=outer_forgetgate,
                                                   alpha=self.W_outer_ln_forgetgate,
                                                   beta=self.b_outer_ln_forgetgate)
                outer_cell_input = self.layer_norm(input=outer_cell_input,
                                                   alpha=self.W_outer_ln_cell,
                                                   beta=self.b_outer_ln_cell)

            if self.grad_clipping:
                outer_ingate = theano.gradient.grad_clip(outer_ingate,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                outer_forgetgate = theano.gradient.grad_clip(outer_forgetgate,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
                outer_cell_input = theano.gradient.grad_clip(outer_cell_input,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
            # get gate nonlinear
            outer_ingate = T.nnet.sigmoid(outer_ingate)
            outer_forgetgate = T.nnet.sigmoid(outer_forgetgate)
            outer_cell_input = T.tanh(outer_cell_input)

            outer_cell = outer_forgetgate*outer_cell_previous + outer_ingate*outer_cell_input
            outer_outgate = slice_outer(outer_gates, 3) + outer_cell*slice_outer(scale_outer_cell, 3)

            if self.use_layer_norm:
                outer_outgate = self.layer_norm(input=outer_outgate,
                                                alpha=self.W_outer_ln_outgate,
                                                beta=self.b_outer_ln_outgate)

            if self.grad_clipping:
                outer_outgate = theano.gradient.grad_clip(outer_outgate,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)
            outer_outgate = T.nnet.sigmoid(outer_outgate)

            outer_outcell = outer_cell
            if self.use_layer_norm:
                outer_outcell = self.layer_norm(input=outer_outcell,
                                                alpha=self.W_outer_ln_outcell,
                                                beta=self.b_outer_ln_outcell)

            if self.grad_clipping:
                outer_outcell = theano.gradient.grad_clip(outer_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            outer_hid = outer_outgate*T.tanh(outer_outcell)
            outer_hid = T.dot(outer_hid, self.W_outer_hid_prj)
            return [inner_cell,
                    inner_hid,
                    outer_cell,
                    outer_hid]

        def step_masked(inner_input_n,
                        outer_input_n,
                        mask_input_n,
                        inner_cell_previous,
                        inner_hid_previous,
                        outer_cell_previous,
                        outer_hid_previous,
                        *args):
            inner_cell, inner_hid, outer_cell, outer_hid = step(inner_input_n,
                                                                outer_input_n,
                                                                inner_cell_previous,
                                                                inner_hid_previous,
                                                                outer_cell_previous,
                                                                outer_hid_previous,
                                                                *args)

            inner_cell = T.switch(mask_input_n, inner_cell, inner_cell_previous)
            inner_hid = T.switch(mask_input_n, inner_hid, inner_hid_previous)
            outer_cell = T.switch(mask_input_n, outer_cell, outer_cell_previous)
            outer_hid = T.switch(mask_input_n, outer_hid, outer_hid_previous)
            return [inner_cell, inner_hid, outer_cell, outer_hid]

        # sequence input
        sequences = [inner_input,
                     outer_input,
                     mask_input]
        step_fun = step_masked

        # state init
        ones = T.ones((num_batch, 1))
        inner_cell_init = T.dot(ones, self.inner_cell_init)
        inner_hid_init = T.dot(ones, self.inner_hid_init)
        outer_cell_init = T.dot(ones, self.outer_cell_init)
        outer_hid_init = T.dot(ones, self.outer_hid_init)

        # non sequence
        non_seqs = []
        non_seqs += [W_inner_hid_to_inner_concat,
                     W_inner_cell_to_inner_concat,
                     W_outer_hid_to_inner_concat,
                     b_inner_concat]

        non_seqs += [self.W_inner_cell_to_inner_ingate,
                     self.W_inner_cell_to_inner_forgetgate,
                     self.W_inner_cell_to_inner_cell,
                     self.W_inner_cell_to_inner_outgate]

        non_seqs += [W_outer_hid_to_outer_concat,
                     W_inner_hid_to_outer_in_concat,
                     b_inner_hid_to_outer_in_concat,
                     W_inner_hid_to_outer_hid_concat,
                     b_inner_hid_to_outer_hid_concat,
                     W_inner_hid_to_outer_cell_concat,
                     b_inner_hid_to_outer_cell_concat,
                     W_inner_hid_to_outer_bias_concat,
                     b_inner_hid_to_outer_bias_concat,
                     self.W_outer_hid_prj]

        if self.use_layer_norm:
            non_seqs +=[self.W_inner_ln_ingate,
                        self.W_inner_ln_forgetgate,
                        self.W_inner_ln_cell,
                        self.W_inner_ln_outgate,
                        self.W_inner_ln_outcell,
                        self.b_inner_ln_ingate,
                        self.b_inner_ln_forgetgate,
                        self.b_inner_ln_cell,
                        self.b_inner_ln_outgate,
                        self.b_inner_ln_outcell,]

            non_seqs +=[self.W_outer_ln_ingate,
                        self.W_outer_ln_forgetgate,
                        self.W_outer_ln_cell,
                        self.W_outer_ln_outgate,
                        self.W_outer_ln_outcell,
                        self.b_outer_ln_ingate,
                        self.b_outer_ln_forgetgate,
                        self.b_outer_ln_cell,
                        self.b_outer_ln_outgate,
                        self.b_outer_ln_outcell,]

        [inner_cell_out,
         inner_hid_out,
         outer_cell_out,
         outer_hid_out] = theano.scan(fn=step_fun,
                                      sequences=sequences,
                                      outputs_info=[inner_cell_init,
                                                    inner_hid_init,
                                                    outer_cell_init,
                                                    outer_hid_init],
                                      go_backwards=self.backwards,
                                      truncate_gradient=self.gradient_steps,
                                      non_sequences=non_seqs,
                                      strict=True)[0]

        inner_hid_out = inner_hid_out.dimshuffle(1, 0, 2)
        outer_hid_out = outer_hid_out.dimshuffle(1, 0, 2)
        if self.backwards:
            inner_hid_out = inner_hid_out[:, ::-1]
            outer_hid_out = outer_hid_out[:, ::-1]

        if self.only_return_outer:
            return outer_hid_out
        else:
            return T.concatenate([inner_hid_out, outer_hid_out], axis=-1)

class ExternalHyperLSTMLayer(MergeLayer):
    def __init__(self,
                 inner_incoming,
                 outer_incoming,
                 num_factor_units,
                 num_outer_units,
                 ingate=Gate(),
                 forgetgate=Gate(b=init.Constant(1.)),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 gating_nonlinearity=None,
                 outer_cell_init=init.Constant(0.),
                 outer_hid_init=init.Constant(0.),
                 dropout_ratio=0.2,
                 use_layer_norm=False,
                 weight_noise=0.0,
                 backwards=False,
                 learn_init=False,
                 peepholes=False,
                 gradient_steps=-1,
                 grad_clipping=1,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        incomings = [inner_incoming,
                     outer_incoming]

        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        self.outer_hid_init_incoming_index = -1
        if isinstance(outer_hid_init, Layer):
            incomings.append(outer_hid_init)
            self.outer_hid_init_incoming_index = len(incomings)-1

        self.outer_cell_init_incoming_index = -1
        if isinstance(outer_cell_init, Layer):
            incomings.append(outer_cell_init)
            self.outer_cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(ExternalHyperLSTMLayer, self).__init__(incomings, **kwargs)

        # for dropout
        self.binomial = RandomStreams(get_rng().randint(1, 2147462579)).binomial
        self.p = dropout_ratio

        # for layer norm
        self.use_layer_norm = use_layer_norm

        # for weight noise
        self.weight_noise = weight_noise
        self.normal = RandomStreams(get_rng().randint(1, 2147462579)).normal

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if gating_nonlinearity is None:
            self.gating_nonlinearity = nonlinearities.identity
        else:
            self.gating_nonlinearity = gating_nonlinearity

        self.learn_init = learn_init
        self.num_factor_units = num_factor_units
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
        inner_input_shape = self.input_shapes[0]
        outer_input_shape = self.input_shapes[1]

        num_inner_inputs = numpy.prod(inner_input_shape[2:])
        num_outer_inputs = numpy.prod(outer_input_shape[2:])

        ###################
        # inner-to-factor #
        ###################
        def add_inner2fact_params(gate, gate_name):
            return (self.add_param(gate.W_hid,
                                   shape=(num_inner_inputs, num_factor_units),
                                   name="V0_in_to_{}".format(gate_name)),
                    self.add_param(gate.b,
                                   shape=(num_factor_units,),
                                   name="b0_in_to_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(gate.W_hid,
                                   shape=(num_inner_inputs, num_factor_units),
                                   name="V0_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b,
                                   shape=(num_factor_units,),
                                   name="b0_hid_to_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(gate.W_hid,
                                   shape=(num_inner_inputs, num_factor_units),
                                   name="V0_bias_{}".format(gate_name)))

        ####ingate####
        (self.V0_in_to_ingate,
         self.b0_in_to_ingate,
         self.V0_hid_to_ingate,
         self.b0_hid_to_ingate,
         self.V0_bias_ingate) = add_inner2fact_params(ingate, 'ingate')

        ####forgetgate####
        (self.V0_in_to_forgetgate,
         self.b0_in_to_forgetgate,
         self.V0_hid_to_forgetgate,
         self.b0_hid_to_forgetgate,
         self.V0_bias_forgetgate) = add_inner2fact_params(forgetgate, 'forgetgate')

        ####cell####
        (self.V0_in_to_cell,
         self.b0_in_to_cell,
         self.V0_hid_to_cell,
         self.b0_hid_to_cell,
         self.V0_bias_cell) = add_inner2fact_params(cell, 'cell')

        ####outgate####
        (self.V0_in_to_outgate,
         self.b0_in_to_outgate,
         self.V0_hid_to_outgate,
         self.b0_hid_to_outgate,
         self.V0_bias_outgate) = add_inner2fact_params(outgate, 'outgate')

        ###################
        # factor-to-outer #
        ###################
        def add_fact2outer_params(gate, gate_name):
            return (self.add_param(gate.W_hid,
                                   shape=(num_factor_units, num_outer_units),
                                   name="V1_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid,
                                   shape=(num_factor_units, num_outer_units),
                                   name="V1_hid_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid,
                                   shape=(num_factor_units, num_outer_units),
                                   name="V1_bias_{}".format(gate_name)))

        ####ingate####
        (self.V1_in_to_ingate,
         self.V1_hid_to_ingate,
         self.V1_bias_ingate) = add_fact2outer_params(ingate, 'ingate')

        ####outgate####
        (self.V1_in_to_forgetgate,
         self.V1_hid_to_forgetgate,
         self.V1_bias_forgetgate) = add_fact2outer_params(forgetgate, 'forgetgate')

        ####cell####
        (self.V1_in_to_cell,
         self.V1_hid_to_cell,
         self.V1_bias_cell) = add_fact2outer_params(cell, 'cell')

        ####outgate####
        (self.V1_in_to_outgate,
         self.V1_hid_to_outgate,
         self.V1_bias_outgate) = add_fact2outer_params(outgate, 'outgate')

        ##############
        # outer loop #
        ##############
        def add_outer_gate_params(gate, gate_name):
            return (self.add_param(gate.W_in,
                                   shape=(num_outer_inputs, num_outer_units),
                                   name="W_outer_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid,
                                   shape=(num_outer_units, num_outer_units),
                                   name="W_outer_hid_to_{}".format(gate_name)),
                    gate.nonlinearity)

        ####ingate####
        (self.W_outer_in_to_ingate,
         self.W_outer_hid_to_ingate,
         self.nonlinearity_outer_ingate) = add_outer_gate_params(ingate, 'ingate')
        if self.use_layer_norm:
            self.alpha_outer_ingate = self.add_param(init.Constant(1.),
                                                     shape=(num_outer_units,),
                                                     name="alpha_outer_ingate")
            self.beta_outer_ingate = self.add_param(init.Constant(0.),
                                                    shape=(num_outer_units,),
                                                    name="beta_outer_ingate")

        ####forgetgate####
        (self.W_outer_in_to_forgetgate,
         self.W_outer_hid_to_forgetgate,
         self.nonlinearity_outer_forgetgate) = add_outer_gate_params(forgetgate, 'forgetgate')
        if self.use_layer_norm:
            self.alpha_outer_forgetgate = self.add_param(init.Constant(1.),
                                                         shape=(num_outer_units,),
                                                         name="alpha_outer_forgetgate")
            self.beta_outer_forgetgate = self.add_param(init.Constant(0.),
                                                        shape=(num_outer_units,),
                                                        name="beta_outer_forgetgate")

        ####cell####
        (self.W_outer_in_to_cell,
         self.W_outer_hid_to_cell,
         self.nonlinearity_outer_cell) = add_outer_gate_params(cell, 'cell')
        if self.use_layer_norm:
            self.alpha_outer_cell = self.add_param(init.Constant(1.),
                                                   shape=(num_outer_units,),
                                                   name="alpha_outer_cell")
            self.beta_outer_cell = self.add_param(init.Constant(0.),
                                                  shape=(num_outer_units,),
                                                  name="beta_outer_cell")

        ####outgate####
        (self.W_outer_in_to_outgate,
         self.W_outer_hid_to_outgate,
         self.nonlinearity_outer_outgate) = add_outer_gate_params(outgate, 'outgate')
        if self.use_layer_norm:
            self.alpha_outer_outgate = self.add_param(init.Constant(1.),
                                                      shape=(num_outer_units,),
                                                      name="alpha_outer_outgate")
            self.beta_outer_outgate = self.add_param(init.Constant(0.),
                                                     shape=(num_outer_units,),
                                                     name="beta_outer_outgate")

        ####peephole####
        if self.peepholes:
            self.W_outer_cell_to_ingate = self.add_param(ingate.W_cell,
                                                         shape=(num_outer_units, ),
                                                         name="W_outer_cell_to_ingate")

            self.W_outer_cell_to_forgetgate = self.add_param(forgetgate.W_cell,
                                                             shape=(num_outer_units, ),
                                                             name="W_outer_cell_to_forgetgate")

            self.W_outer_cell_to_outgate = self.add_param(outgate.W_cell,
                                                          shape=(num_outer_units, ),
                                                          name="W_outer_cell_to_outgate")

        ####layer_norm####
        if self.use_layer_norm:
            self.alpha_outer_outcell = self.add_param(init.Constant(1.),
                                                      shape=(num_outer_units,),
                                                      name="alpha_outer_outcell")
            self.beta_outer_outcell =  self.add_param(init.Constant(0.),
                                                      shape=(num_outer_units,),
                                                      name="beta_outer_outcell",
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
        inner_input = inputs[0]
        outer_input = inputs[1]

        # mask
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # outer hid
        outer_hid_init = None
        if self.outer_hid_init_incoming_index > 0:
            outer_hid_init = inputs[self.outer_hid_init_incoming_index]

        # outer cell
        outer_cell_init = None
        if self.outer_cell_init_incoming_index > 0:
            outer_cell_init = inputs[self.outer_cell_init_incoming_index]

        if inner_input.ndim > 3:
            inner_input = T.flatten(inner_input, 3)

        if outer_input.ndim > 3:
            outer_input = T.flatten(outer_input, 3)

        inner_input = inner_input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = inner_input.shape

        outer_input = outer_input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = outer_input.shape

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
            outer_input = T.dot(outer_input, W_outer_in_stacked)
        else:
            outer_input = outer_input

        # slice for outer
        def slice_outer(x, n):
            return x[:, n*self.num_outer_units:(n+1)*self.num_outer_units]

        # slice for factor
        def slice_factor(x, n):
            return x[:, n*self.num_factor_units:(n+1)*self.num_factor_units]

        # if using dropout
        if deterministic:
            self.using_dropout = False
        else:
            self.using_dropout = True

        outer_cell_mask = self.binomial((num_batch, self.num_outer_units),
                                        p=T.constant(1) - self.p,
                                        dtype=floatX)

        # step function
        def step(inner_input,
                 outer_input,
                 outer_cell_previous,
                 outer_hid_previous,
                 *args):

            ###################
            # inner to factor #
            ###################
            fact_in = T.dot(inner_input, V0_in_stacked) + b0_in_stacked
            fact_hid = T.dot(inner_input, V0_hid_stacked) + b0_hid_stacked
            fact_bias = T.dot(inner_input, V0_bias_stacked)

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

            scale_outer_in = self.gating_nonlinearity(scale_outer_in)
            scale_outer_hid = self.gating_nonlinearity(scale_outer_hid)

            ##############
            # outer loop #
            ##############
            if not self.precompute_input:
                outer_input = T.dot(outer_input, W_outer_in_stacked)
            outer_gates = scale_outer_in*outer_input
            outer_gates += scale_outer_hid*T.dot(outer_hid_previous, W_outer_hid_stacked)
            outer_gates += outer_bias

            # get gate slices
            outer_ingate = slice_outer(outer_gates, 0)
            outer_forgetgate = slice_outer(outer_gates, 1)
            outer_cell_input = slice_outer(outer_gates, 2)
            outer_outgate = slice_outer(outer_gates, 3)

            # get peepholes
            if self.peepholes:
                outer_ingate += outer_cell_previous*self.W_outer_cell_to_ingate
                outer_forgetgate += outer_cell_previous*self.W_outer_cell_to_forgetgate

            if self.grad_clipping:
                outer_ingate = theano.gradient.grad_clip(outer_ingate,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                outer_forgetgate = theano.gradient.grad_clip(outer_forgetgate,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
                outer_cell_input = theano.gradient.grad_clip(outer_cell_input,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)

            if self.use_layer_norm:
                outer_ingate = self.layer_norm(input=outer_ingate,
                                               alpha=self.alpha_outer_ingate,
                                               beta=self.beta_outer_ingate)
                outer_forgetgate = self.layer_norm(input=outer_forgetgate,
                                                   alpha=self.alpha_outer_forgetgate,
                                                   beta=self.beta_outer_forgetgate)
                outer_cell_input = self.layer_norm(input=outer_cell_input,
                                                   alpha=self.alpha_outer_cell,
                                                   beta=self.beta_outer_cell)
                outer_outgate = self.layer_norm(input=outer_outgate,
                                                alpha=self.alpha_outer_outgate,
                                                beta=self.beta_outer_outgate)

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

            outer_cell = outer_forgetgate*outer_cell_previous + outer_ingate*outer_cell_input

            if self.peepholes:
                outer_outgate += outer_cell*self.W_outer_cell_to_outgate

            if self.grad_clipping:
                outer_outgate = theano.gradient.grad_clip(outer_outgate,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)
            if self.use_layer_norm:
                outer_outgate = self.layer_norm(input=outer_outgate,
                                                alpha=self.alpha_outer_outgate,
                                                beta=self.beta_outer_outgate)

            outer_outgate = self.nonlinearity_outer_outgate(outer_outgate)

            if self.use_layer_norm:
                _cell = self.layer_norm(input=outer_cell,
                                        alpha=self.alpha_outer_outcell,
                                        beta=self.beta_outer_outcell)
            else:
                _cell = outer_cell

            if self.grad_clipping:
                _cell = theano.gradient.grad_clip(_cell,
                                                  -self.grad_clipping,
                                                  self.grad_clipping)

            outer_hid = outer_outgate*self.nonlinearity(_cell)
            return [outer_cell, outer_hid]

        def step_masked(inner_input,
                        outer_input,
                        mask_n,
                        outer_cell_previous,
                        outer_hid_previous,
                        *args):
            outer_cell, outer_hid = step(inner_input,
                                         outer_input,
                                         outer_cell_previous,
                                         outer_hid_previous,
                                         *args)

            outer_cell = T.switch(mask_n, outer_cell, outer_cell_previous)
            outer_hid = T.switch(mask_n, outer_hid, outer_hid_previous)
            return [outer_cell, outer_hid]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [inner_input, outer_input, mask]
            step_fun = step_masked
        else:
            sequences = [inner_input, outer_input]
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.outer_cell_init, Layer):
            outer_cell_init = T.dot(ones, self.outer_cell_init)

        if not isinstance(self.outer_hid_init, Layer):
            outer_hid_init = T.dot(ones, self.outer_hid_init)

        non_seqs = [outer_cell_mask]
        if not self.precompute_input:
            non_seqs += [W_outer_in_stacked]

        non_seqs += [W_outer_hid_stacked,
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
            non_seqs += [self.W_outer_cell_to_ingate,
                         self.W_outer_cell_to_forgetgate,
                         self.W_outer_cell_to_outgate]

        if self.use_layer_norm:
            non_seqs +=[self.alpha_outer_ingate,
                        self.alpha_outer_forgetgate,
                        self.alpha_outer_cell,
                        self.alpha_outer_outgate,
                        self.alpha_outer_outcell,
                        self.beta_outer_ingate,
                        self.beta_outer_forgetgate,
                        self.beta_outer_cell,
                        self.beta_outer_outgate,
                        self.beta_outer_outcell,]
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            outer_cell_out, outer_hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[outer_cell_init, outer_hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            outer_cell_out, outer_hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[outer_cell_init, outer_hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        if self.only_return_final:
            outer_hid_out = outer_hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            outer_hid_out = outer_hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                outer_hid_out = outer_hid_out[:, ::-1]
        return outer_hid_out

# class FactorScalingHyperLSTMLayer(MergeLayer):
#     def __init__(self,
#                  inner_incoming,
#                  outer_incoming,
#                  mask_incoming,
#                  num_inner_units,
#                  num_factor_units,
#                  num_outer_units,
#                  inner_cell_init=init.Constant(0.),
#                  inner_hid_init=init.Constant(0.),
#                  outer_cell_init=init.Constant(0.),
#                  outer_hid_init=init.Constant(0.),
#                  dropout_ratio=0.2,
#                  use_layer_norm=False,
#                  weight_noise=0.0,
#                  backwards=False,
#                  learn_init=False,
#                  gradient_steps=-1,
#                  grad_clipping=1,
#                  only_return_outer=False,
#                  **kwargs):
#
#         # set input layers
#         incomings = [inner_incoming,
#                      outer_incoming,
#                      mask_incoming]
#
#         # init states
#         self.inner_hid_init_incoming_index = -1
#         if isinstance(inner_hid_init, Layer):
#             incomings.append(inner_hid_init)
#             self.inner_hid_init_incoming_index = len(incomings)-1
#
#         self.inner_cell_init_incoming_index = -1
#         if isinstance(inner_cell_init, Layer):
#             incomings.append(inner_cell_init)
#             self.inner_cell_init_incoming_index = len(incomings)-1
#
#         self.outer_hid_init_incoming_index = -1
#         if isinstance(outer_hid_init, Layer):
#             incomings.append(outer_hid_init)
#             self.outer_hid_init_incoming_index = len(incomings)-1
#
#         self.outer_cell_init_incoming_index = -1
#         if isinstance(outer_cell_init, Layer):
#             incomings.append(outer_cell_init)
#             self.outer_cell_init_incoming_index = len(incomings)-1
#
#         # initialize
#         super(FactorScalingHyperLSTMLayer, self).__init__(incomings, **kwargs)
#
#         # set options
#         self.num_inner_units = num_inner_units
#         self.num_factor_units = num_factor_units
#         self.num_outer_units = num_outer_units
#         self.learn_init = learn_init
#         self.backwards = backwards
#         self.gradient_steps = gradient_steps
#         self.grad_clipping = grad_clipping
#         self.only_return_outer = only_return_outer
#
#         # for dropout
#         self.binomial = RandomStreams(get_rng().randint(1, 2147462579)).binomial
#         self.p = dropout_ratio
#
#         # for layer norm
#         self.use_layer_norm = use_layer_norm
#
#         # for weight noise
#         self.weight_noise = weight_noise
#         self.normal = RandomStreams(get_rng().randint(1, 2147462579)).normal
#
#         # get size
#         inner_input_shape = self.input_shapes[0]
#         outer_input_shape = self.input_shapes[1]
#         num_inner_inputs = numpy.prod(inner_input_shape[2:])
#         num_outer_inputs = numpy.prod(outer_input_shape[2:])
#
#         ##############
#         # inner loop #
#         ##############
#         def add_inner_gate_params(gate_name,
#                                   cell_trainable=True,
#                                   bias_const=0.0):
#             return (#### input-to-hidden ####
#                     self.add_param(init.Orthogonal(),
#                                    shape=(num_inner_inputs, num_inner_units),
#                                    name="W_inner_in_to_inner_{}".format(gate_name)),
#                     #### hidden-to-hidden ####
#                     self.add_param(init.Orthogonal(),
#                                    shape=(num_inner_units, num_inner_units),
#                                    name="W_inner_hid_to_inner_{}".format(gate_name)),
#                     #### hidden-to-hidden ####
#                     self.add_param(init.Uniform(0.1) if cell_trainable else init.Constant(0.0),
#                                    shape=(num_inner_units, num_inner_units),
#                                    name="W_inner_cell_to_inner_{}".format(gate_name),
#                                    trainable=cell_trainable),
#                     #### hidden-to-hidden ####
#                     self.add_param(init.Orthogonal(),
#                                    shape=(num_outer_units, num_inner_units),
#                                    name="W_outer_hid_to_inner_{}".format(gate_name)),
#                     #### bias ####
#                     self.add_param(init.Constant(bias_const),
#                                    shape=(num_inner_units,),
#                                    name="b_inner_{}".format(gate_name),
#                                    regularizable=False))
#
#         ####ingate####
#         (self.W_inner_in_to_inner_ingate,
#          self.W_inner_hid_to_inner_ingate,
#          self.W_inner_cell_to_inner_ingate,
#          self.W_outer_cell_to_inner_ingate,
#          self.b_inner_ingate) = add_inner_gate_params('ingate')
#         if self.use_layer_norm:
#             self.alpha_inner_ingate = self.add_param(init.Constant(1.),
#                                                      shape=(num_inner_units,),
#                                                      name="alpha_inner_ingate")
#             self.beta_inner_ingate = self.add_param(init.Constant(0.),
#                                                     shape=(num_inner_units,),
#                                                     name="beta_inner_ingate")
#
#         ####forgetgate#####
#         (self.W_inner_in_to_inner_forgetgate,
#          self.W_inner_hid_to_inner_forgetgate,
#          self.W_inner_cell_to_inner_forgetgate,
#          self.W_outer_cell_to_inner_forgetgate,
#          self.b_inner_forgetgate) = add_inner_gate_params('forgetgate', bias_const=1.0)
#         if self.use_layer_norm:
#             self.alpha_inner_forgetgate = self.add_param(init.Constant(1.),
#                                                          shape=(num_inner_units,),
#                                                          name="alpha_inner_forgetgate")
#             self.beta_inner_forgetgate = self.add_param(init.Constant(0.),
#                                                         shape=(num_inner_units,),
#                                                         name="beta_inner_forgetgate")
#
#         ####cell#####
#         (self.W_inner_in_to_inner_cell,
#          self.W_inner_hid_to_inner_cell,
#          self.W_inner_cell_to_inner_cell,
#          self.W_outer_cell_to_inner_cell,
#          self.b_inner_cell) = add_inner_gate_params('cell', cell_trainable=False)
#         if self.use_layer_norm:
#             self.alpha_inner_cell = self.add_param(init.Constant(1.),
#                                                    shape=(num_inner_units,),
#                                                    name="alpha_inner_cell")
#             self.beta_inner_cell = self.add_param(init.Constant(0.),
#                                                   shape=(num_inner_units,),
#                                                   name="beta_inner_cell")
#
#         ####outgate#####
#         (self.W_inner_in_to_inner_outgate,
#          self.W_inner_hid_to_inner_outgate,
#          self.W_inner_cell_to_inner_outgate,
#          self.W_outer_cell_to_inner_outgate,
#          self.b_inner_outgate) = add_inner_gate_params('outgate')
#         if self.use_layer_norm:
#             self.alpha_inner_outgate = self.add_param(init.Constant(1.),
#                                                       shape=(num_inner_units,),
#                                                       name="alpha_inner_outgate")
#             self.beta_inner_outgate = self.add_param(init.Constant(0.),
#                                                      shape=(num_inner_units,),
#                                                      name="beta_inner_outgate")
#
#         ####out cell#####
#         if self.use_layer_norm:
#             self.alpha_inner_outcell = self.add_param(init.Constant(1.),
#                                                       shape=(num_inner_units,),
#                                                       name="alpha_inner_outcell")
#             self.beta_inner_outcell =  self.add_param(init.Constant(0.),
#                                                       shape=(num_inner_units,),
#                                                       name="beta_inner_outcell",
#                                                       regularizable=False)
#
#         ####init_cell####
#         if isinstance(inner_cell_init, Layer):
#             self.inner_cell_init = inner_cell_init
#         else:
#             self.inner_cell_init = self.add_param(inner_cell_init,
#                                                   shape=(1, num_inner_units),
#                                                   name="inner_cell_init",
#                                                   trainable=learn_init,
#                                                   regularizable=False)
#
#         ####init_hidden####
#         if isinstance(inner_hid_init, Layer):
#             self.inner_hid_init = inner_hid_init
#         else:
#             self.inner_hid_init = self.add_param(inner_hid_init,
#                                                  shape=(1, num_inner_units),
#                                                  name="inner_hid_init",
#                                                  trainable=learn_init,
#                                                  regularizable=False)
#
#         ###################
#         # inner-to-factor #
#         ###################
#         def add_inner2fact_params(gate_name, cell_trainable=True):
#             return (#### inner-to-fact input ####
#                     self.add_param(init.Constant(0.0),
#                                    shape=(num_inner_units, num_factor_units),
#                                    name="W_fact_in_{}".format(gate_name)),
#                     self.add_param(init.Constant(1.0),
#                                    shape=(num_factor_units,),
#                                    name="b_fact_in_{}".format(gate_name),
#                                    regularizable=False),
#                     #### inner-to-fact hidden ####
#                     self.add_param(init.Constant(0.0),
#                                    shape=(num_inner_units, num_factor_units),
#                                    name="W_fact_hid_{}".format(gate_name)),
#                     self.add_param(init.Constant(1.0),
#                                    shape=(num_factor_units,),
#                                    name="b_fact_hid_{}".format(gate_name),
#                                    regularizable=False),
#                     #### inner-to-fact cell ####
#                     self.add_param(init.Constant(0.0),
#                                    shape=(num_inner_units, num_factor_units),
#                                    name="W_fact_cell_{}".format(gate_name)),
#                     self.add_param(init.Constant(1.0),
#                                    shape=(num_factor_units,),
#                                    name="b_fact_cell_{}".format(gate_name),
#                                    regularizable=False),
#                     #### inner-to-fact bias ####
#                     self.add_param(init.Normal(0.01),
#                                    shape=(num_inner_units, num_factor_units),
#                                    name="W_fact_b_{}".format(gate_name)))
#         ####ingate####
#         (self.W_fact_in_ingate, self.b_fact_in_ingate,
#          self.W_fact_hid_ingate, self.b_fact_hid_ingate,
#          self.W_fact_cell_ingate, self.b_fact_cell_ingate,
#          self.W_fact_b_ingate) = add_inner2fact_params('ingate')
#
#         ####forgetgate####
#         (self.W_fact_in_forgetgate, self.b_fact_in_forgetgate,
#          self.W_fact_hid_forgetgate, self.b_fact_hid_forgetgate,
#          self.W_fact_cell_forgetgate, self.b_fact_cell_forgetgate,
#          self.W_fact_b_forgetgate) = add_inner2fact_params('forgetgate')
#
#         ####cell####
#         (self.W_fact_in_cell, self.b_fact_in_cell,
#          self.W_fact_hid_cell, self.b_fact_hid_cell,
#          self.W_fact_cell_cell, self.b_fact_cell_cell,
#          self.W_fact_b_cell) = add_inner2fact_params('cell')
#
#         ####outgate####
#         (self.W_fact_in_outgate, self.b_fact_in_outgate,
#          self.W_fact_hid_outgate, self.b_fact_hid_outgate,
#          self.W_fact_cell_outgate, self.b_fact_cell_outgate,
#          self.W_fact_b_outgate) = add_inner2fact_params('outgate')
#
#         ###################
#         # factor-to-outer #
#         ###################
#         def add_fact2outer_params(gate_name):
#             return (#### fact-to-outer input ####
#                     self.add_param(init.Constant(1.0/num_factor_units),
#                                    shape=(num_factor_units, num_outer_units),
#                                    name="W_fact_to_in_{}".format(gate_name)),
#                     #### fact-to-outer hidden ####
#                     self.add_param(init.Constant(1.0/num_factor_units),
#                                    shape=(num_factor_units, num_outer_units),
#                                    name="W_fact_to_hid_{}".format(gate_name)),
#                     #### fact-to-outer bias ####
#                     self.add_param(init.Constant(0.0),
#                                    shape=(num_factor_units, num_outer_units),
#                                    name="W_fact_to_b_{}".format(gate_name)))
#
#         ####ingate####
#         (self.W_fact_to_in_ingate,
#          self.W_fact_to_hid_ingate,
#          self.W_fact_to_b_ingate) = add_fact2outer_params('ingate')
#
#         ####outgate####
#         (self.W_fact_to_in_forgetgate,
#          self.W_fact_to_hid_forgetgate,
#          self.W_fact_to_b_forgetgate) = add_fact2outer_params('forgetgate')
#
#         ####cell####
#         (self.W_fact_to_in_cell,
#          self.W_fact_to_hid_cell,
#          self.W_fact_to_b_cell) = add_fact2outer_params('cell')
#
#         ####outgate####
#         (self.W_fact_to_in_outgate,
#          self.W_fact_to_hid_outgate,
#          self.W_fact_to_b_outgate) = add_fact2outer_params('outgate')
#
#
#         ##############
#         # outer loop #
#         ##############
#         def add_outer_gate_params(gate_name):
#             return (#### outer input-to-hidden ####
#                     self.add_param(init.Orthogonal(0.1),
#                                    shape=(num_outer_inputs, num_outer_units),
#                                    name="W_outer_in_to_{}".format(gate_name)),
#                     #### outer hidden-to-hidden ####
#                     self.add_param(init.Orthogonal(0.1),
#                                    shape=(num_outer_units, num_outer_units),
#                                    name="W_outer_hid_to_{}".format(gate_name)))
#
#         ####ingate####
#         (self.W_outer_in_to_ingate,
#          self.W_outer_hid_to_ingate) = add_outer_gate_params('ingate')
#         self.W_outer_cell_to_ingate = self.add_param(init.Uniform(0.1),
#                                                      shape=(num_outer_units, ),
#                                                      name="W_outer_cell_to_ingate")
#         if self.use_layer_norm:
#             self.alpha_outer_ingate = self.add_param(init.Constant(1.),
#                                                      shape=(num_outer_units,),
#                                                      name="alpha_outer_ingate")
#             self.beta_outer_ingate = self.add_param(init.Constant(0.),
#                                                     shape=(num_outer_units,),
#                                                     name="beta_outer_ingate")
#
#         ####forgetgate####
#         (self.W_outer_in_to_forgetgate,
#          self.W_outer_hid_to_forgetgate) = add_outer_gate_params('forgetgate')
#         self.W_outer_cell_to_forgetgate = self.add_param(init.Uniform(0.1),
#                                                          shape=(num_outer_units, ),
#                                                          name="W_outer_cell_to_forgetgate")
#         if self.use_layer_norm:
#             self.alpha_outer_forgetgate = self.add_param(init.Constant(1.),
#                                                          shape=(num_outer_units,),
#                                                          name="alpha_outer_forgetgate")
#             self.beta_outer_forgetgate = self.add_param(init.Constant(0.),
#                                                         shape=(num_outer_units,),
#                                                         name="beta_outer_forgetgate")
#
#         ####cell####
#         (self.W_outer_in_to_cell,
#          self.W_outer_hid_to_cell) = add_outer_gate_params('cell')
#         if self.use_layer_norm:
#             self.alpha_outer_cell = self.add_param(init.Constant(1.),
#                                                    shape=(num_outer_units,),
#                                                    name="alpha_outer_cell")
#             self.beta_outer_cell = self.add_param(init.Constant(0.),
#                                                   shape=(num_outer_units,),
#                                                   name="beta_outer_cell")
#
#         ####outgate####
#         (self.W_outer_in_to_outgate,
#          self.W_outer_hid_to_outgate) = add_outer_gate_params('outgate')
#         self.W_outer_cell_to_outgate = self.add_param(init.Uniform(0.1),
#                                                       shape=(num_outer_units, ),
#                                                       name="W_outer_cell_to_outgate")
#         if self.use_layer_norm:
#             self.alpha_outer_outgate = self.add_param(init.Constant(1.),
#                                                       shape=(num_outer_units,),
#                                                       name="alpha_outer_outgate")
#             self.beta_outer_outgate = self.add_param(init.Constant(0.),
#                                                      shape=(num_outer_units,),
#                                                      name="beta_outer_outgate")
#
#         ####out cell####
#         if self.use_layer_norm:
#             self.alpha_outer_outcell = self.add_param(init.Constant(1.),
#                                                       shape=(num_outer_units,),
#                                                       name="alpha_outer_outcell")
#             self.beta_outer_outcell =  self.add_param(init.Constant(0.),
#                                                       shape=(num_outer_units,),
#                                                       name="beta_outer_outcell",
#                                                       regularizable=False)
#         ####init_cell####
#         if isinstance(outer_cell_init, Layer):
#             self.outer_cell_init = outer_cell_init
#         else:
#             self.outer_cell_init = self.add_param(outer_cell_init,
#                                                   shape=(1, num_outer_units),
#                                                   name="outer_cell_init",
#                                                   trainable=learn_init,
#                                                   regularizable=False)
#
#         ####init_hid####
#         if isinstance(outer_hid_init, Layer):
#             self.outer_hid_init = outer_hid_init
#         else:
#             self.outer_hid_init = self.add_param(outer_hid_init,
#                                                  shape=(1, num_outer_units),
#                                                  name="outer_hid_init",
#                                                  trainable=learn_init,
#                                                  regularizable=False)
#
#     def layer_norm(self, input, alpha, beta):
#         output = (input - T.mean(input, axis=1, keepdims=True))/(T.sqrt(T.var(input, axis=1, keepdims=True)) + eps)
#         output = alpha[None, :]*output + beta[None, :]
#         return output
#
#     def get_output_shape_for(self, input_shapes):
#         input_shape = input_shapes[0]
#         if self.only_return_outer:
#             num_outputs = self.num_outer_units
#         else:
#             num_outputs = self.num_inner_units + self.num_outer_units
#
#         return input_shape[0], input_shape[1], num_outputs
#
#     def get_output_for(self, inputs, deterministic=False, **kwargs):
#         # input
#         inner_input = inputs[0]
#         outer_input = inputs[1]
#         mask_input = inputs[2]
#
#         # inner hid
#         inner_hid_init = None
#         if self.inner_hid_init_incoming_index > 0:
#             inner_hid_init = inputs[self.inner_hid_init_incoming_index]
#
#         # inner cell
#         inner_cell_init = None
#         if self.inner_cell_init_incoming_index > 0:
#             inner_cell_init = inputs[self.inner_cell_init_incoming_index]
#
#         # outer hid
#         outer_hid_init = None
#         if self.outer_hid_init_incoming_index > 0:
#             outer_hid_init = inputs[self.outer_hid_init_incoming_index]
#
#         # outer cell
#         outer_cell_init = None
#         if self.outer_cell_init_incoming_index > 0:
#             outer_cell_init = inputs[self.outer_cell_init_incoming_index]
#
#         if inner_input.ndim > 3:
#             inner_input = T.flatten(inner_input, 3)
#         if outer_input.ndim > 3:
#             outer_input = T.flatten(outer_input, 3)
#
#         inner_input = inner_input.dimshuffle(1, 0, 2)
#         seq_len, num_batch, num_inner_inputs = inner_input.shape
#
#         outer_input = outer_input.dimshuffle(1, 0, 2)
#         seq_len, num_batch, num_outer_inputs = outer_input.shape
#
#         mask_input = mask_input.dimshuffle(1, 0, 'x')
#
#         ##############
#         # inner loop #
#         ##############
#         #### input-to-hidden ####
#         W_inner_in_concat = T.concatenate([self.W_inner_in_to_ingate,
#                                            self.W_inner_in_to_forgetgate,
#                                            self.W_inner_in_to_cell,
#                                            self.W_inner_in_to_outgate], axis=1)
#
#         #### hidden-to-hidden ####
#         W_inner_hid_concat = T.concatenate([self.W_inner_hid_to_ingate,
#                                             self.W_inner_hid_to_forgetgate,
#                                             self.W_inner_hid_to_cell,
#                                             self.W_inner_hid_to_outgate], axis=1)
#
#         #### bias ####
#         b_inner_concat = T.concatenate([self.b_inner_ingate,
#                                         self.b_inner_forgetgate,
#                                         self.b_inner_cell,
#                                         self.b_inner_outgate], axis=0)
#
#         ###################
#         # inner to factor #
#         ###################
#         #### inner-to-fact input ####
#         W_fact_in_concat = T.concatenate([self.W_fact_in_ingate,
#                                           self.W_fact_in_forgetgate,
#                                           self.W_fact_in_cell,
#                                           self.W_fact_in_outgate], axis=1)
#         b_fact_in_concat = T.concatenate([self.b_fact_in_ingate,
#                                           self.b_fact_in_forgetgate,
#                                           self.b_fact_in_cell,
#                                           self.b_fact_in_outgate], axis=0)
#
#         #### inner-to-fact hidden ####
#         W_fact_hid_concat = T.concatenate([self.W_fact_hid_ingate,
#                                            self.W_fact_hid_forgetgate,
#                                            self.W_fact_hid_cell,
#                                            self.W_fact_hid_outgate], axis=1)
#         b_fact_hid_concat = T.concatenate([self.b_fact_hid_ingate,
#                                            self.b_fact_hid_forgetgate,
#                                            self.b_fact_hid_cell,
#                                            self.b_fact_hid_outgate], axis=0)
#
#         ####bias####
#         W_fact_b_concat = T.concatenate([self.W_fact_b_ingate,
#                                          self.W_fact_b_forgetgate,
#                                          self.W_fact_b_cell,
#                                          self.W_fact_b_outgate], axis=1)
#
#         ##############
#         # outer loop #
#         ##############
#         #### input-to-hidden ####
#         W_outer_in_concat = T.concatenate([self.W_outer_in_to_ingate,
#                                            self.W_outer_in_to_forgetgate,
#                                            self.W_outer_in_to_cell,
#                                            self.W_outer_in_to_outgate], axis=1)
#
#         #### hidden-to-hidden ####
#         W_outer_hid_concat = T.concatenate([self.W_outer_hid_to_ingate,
#                                             self.W_outer_hid_to_forgetgate,
#                                             self.W_outer_hid_to_cell,
#                                             self.W_outer_hid_to_outgate], axis=1)
#
#         # pre-compute inner/outer input
#         inner_input = T.dot(inner_input, W_inner_in_concat) + b_inner_concat
#         outer_input = T.dot(outer_input, W_outer_in_concat)
#
#         # slice for inner
#         def slice_inner(x, n):
#             return x[:, n*self.num_inner_units:(n+1)*self.num_inner_units]
#
#         # slice for outer
#         def slice_outer(x, n):
#             return x[:, n*self.num_outer_units:(n+1)*self.num_outer_units]
#
#         # slice for factor
#         def slice_factor(x, n):
#             return x[:, n*self.num_factor_units:(n+1)*self.num_factor_units]
#
#         # if using dropout
#         if deterministic:
#             self.using_dropout = False
#         else:
#             self.using_dropout = True
#
#         inner_cell_mask = self.binomial((num_batch, self.num_inner_units),
#                                         p=T.constant(1) - self.p,
#                                         dtype=floatX)
#
#         outer_cell_mask = self.binomial((num_batch, self.num_outer_units),
#                                         p=T.constant(1) - self.p,
#                                         dtype=floatX)
#
#         # step function
#         def step(inner_input_n,
#                  outer_input_n,
#                  inner_cell_previous,
#                  inner_hid_previous,
#                  outer_cell_previous,
#                  outer_hid_previous,
#                  *args):
#             ##############
#             # inner loop #
#             ##############
#             inner_gates = inner_input_n
#             inner_gates += T.dot(inner_hid_previous, W_inner_hid_concat)
#
#             inner_ingate = slice_inner(inner_gates, 0) + inner_cell_previous*self.W_inner_cell_to_ingate
#             inner_forgetgate = slice_inner(inner_gates, 1) + inner_cell_previous*self.W_inner_cell_to_forgetgate
#             inner_cell_input = slice_inner(inner_gates, 2)
#
#             if self.grad_clipping:
#                 inner_ingate = theano.gradient.grad_clip(inner_ingate,
#                                                          -self.grad_clipping,
#                                                          self.grad_clipping)
#                 inner_forgetgate = theano.gradient.grad_clip(inner_forgetgate,
#                                                              -self.grad_clipping,
#                                                              self.grad_clipping)
#                 inner_cell_input = theano.gradient.grad_clip(inner_cell_input,
#                                                              -self.grad_clipping,
#                                                              self.grad_clipping)
#
#             if self.use_layer_norm:
#                 inner_ingate = self.layer_norm(input=inner_ingate,
#                                                alpha=self.alpha_inner_ingate,
#                                                beta=self.beta_inner_ingate)
#                 inner_forgetgate = self.layer_norm(input=inner_forgetgate,
#                                                    alpha=self.alpha_inner_forgetgate,
#                                                    beta=self.beta_inner_forgetgate)
#                 inner_cell_input = self.layer_norm(input=inner_cell_input,
#                                                    alpha=self.alpha_inner_cell,
#                                                    beta=self.beta_inner_cell)
#
#             inner_ingate = T.nnet.sigmoid(inner_ingate)
#             inner_forgetgate = T.nnet.sigmoid(inner_forgetgate)
#             inner_cell_input = T.tanh(inner_cell_input)
#
#             if self.using_dropout==False or self.p == 0:
#                 inner_cell_input = inner_cell_input
#             else:
#                 one = T.constant(1)
#                 retain_prob = one - self.p
#                 inner_cell_input /= retain_prob
#                 inner_cell_input = inner_cell_input*inner_cell_mask
#
#             inner_cell = inner_forgetgate*inner_cell_previous + inner_ingate*inner_cell_input
#             inner_outgate = slice_inner(inner_gates, 3) + inner_cell*self.W_inner_cell_to_outgate
#
#             if self.grad_clipping:
#                 inner_outgate = theano.gradient.grad_clip(inner_outgate,
#                                                           -self.grad_clipping,
#                                                           self.grad_clipping)
#             if self.use_layer_norm:
#                 inner_outgate = self.layer_norm(input=inner_outgate,
#                                                 alpha=self.alpha_inner_outgate,
#                                                 beta=self.beta_inner_outgate)
#
#             inner_outgate = T.nnet.sigmoid(inner_outgate)
#
#             if self.use_layer_norm:
#                 _cell = self.layer_norm(input=inner_cell,
#                                         alpha=self.alpha_inner_outcell,
#                                         beta=self.beta_inner_outcell)
#             else:
#                 _cell = inner_cell
#
#             if self.grad_clipping:
#                 _cell = theano.gradient.grad_clip(_cell,
#                                                   -self.grad_clipping,
#                                                   self.grad_clipping)
#
#             # update inner hidden
#             inner_hid = inner_outgate*T.tanh(_cell)
#
#             ###################
#             # inner to factor #
#             ###################
#             fact_in = T.dot(inner_hid, W_fact_in_concat) + b_fact_in_concat
#             fact_hid = T.dot(inner_hid, W_fact_hid_concat) + b_fact_hid_concat
#             fact_bias = T.dot(inner_hid, W_fact_b_concat)
#
#             ###################
#             # factor to outer #
#             ###################
#             scale_outer_in = T.concatenate([T.dot(slice_factor(fact_in, 0), self.W_fact_to_in_ingate),
#                                             T.dot(slice_factor(fact_in, 1), self.W_fact_to_in_forgetgate),
#                                             T.dot(slice_factor(fact_in, 2), self.W_fact_to_in_cell),
#                                             T.dot(slice_factor(fact_in, 3), self.W_fact_to_in_outgate)],
#                                            axis=1)
#             scale_outer_hid = T.concatenate([T.dot(slice_factor(fact_hid, 0), self.W_fact_to_hid_ingate),
#                                              T.dot(slice_factor(fact_hid, 1), self.W_fact_to_hid_forgetgate),
#                                              T.dot(slice_factor(fact_hid, 2), self.W_fact_to_hid_cell),
#                                              T.dot(slice_factor(fact_hid, 3), self.W_fact_to_hid_outgate)],
#                                             axis=1)
#
#             outer_bias = T.concatenate([T.dot(slice_factor(fact_bias, 0), self.W_fact_to_b_ingate),
#                                         T.dot(slice_factor(fact_bias, 1), self.W_fact_to_b_forgetgate),
#                                         T.dot(slice_factor(fact_bias, 2), self.W_fact_to_b_cell),
#                                         T.dot(slice_factor(fact_bias, 3), self.W_fact_to_b_outgate)],
#                                        axis=1)
#
#             scale_outer_in = self.scale_nonlinearity(scale_outer_in)
#             scale_outer_hid = self.scale_nonlinearity(scale_outer_hid)
#
#             ##############
#             # outer loop #
#             ##############
#             outer_gates = scale_outer_in*outer_input_n
#             outer_gates += scale_outer_hid*T.dot(outer_hid_previous, W_outer_hid_concat)
#             outer_gates += outer_bias
#
#             outer_ingate = slice_outer(outer_gates, 0) + outer_cell_previous*self.W_outer_cell_to_ingate
#             outer_forgetgate = slice_outer(outer_gates, 1) + outer_cell_previous*self.W_outer_cell_to_forgetgate
#             outer_cell_input = slice_outer(outer_gates, 2)
#
#             if self.grad_clipping:
#                 outer_ingate = theano.gradient.grad_clip(outer_ingate,
#                                                          -self.grad_clipping,
#                                                          self.grad_clipping)
#                 outer_forgetgate = theano.gradient.grad_clip(outer_forgetgate,
#                                                              -self.grad_clipping,
#                                                              self.grad_clipping)
#                 outer_cell_input = theano.gradient.grad_clip(outer_cell_input,
#                                                              -self.grad_clipping,
#                                                              self.grad_clipping)
#
#             if self.use_layer_norm:
#                 outer_ingate = self.layer_norm(input=outer_ingate,
#                                                alpha=self.alpha_outer_ingate,
#                                                beta=self.beta_outer_ingate)
#                 outer_forgetgate = self.layer_norm(input=outer_forgetgate,
#                                                    alpha=self.alpha_outer_forgetgate,
#                                                    beta=self.beta_outer_forgetgate)
#                 outer_cell_input = self.layer_norm(input=outer_cell_input,
#                                                    alpha=self.alpha_outer_cell,
#                                                    beta=self.beta_outer_cell)
#
#             # get gate nonlinear
#             outer_ingate = T.nnet.sigmoid(outer_ingate)
#             outer_forgetgate = T.nnet.sigmoid(outer_forgetgate)
#             outer_cell_input = T.tanh(outer_cell_input)
#
#             # drop out
#             if self.using_dropout==False or self.p == 0:
#                 outer_cell_input = outer_cell_input
#             else:
#                 one = T.constant(1)
#                 retain_prob = one - self.p
#                 outer_cell_input /= retain_prob
#                 outer_cell_input = outer_cell_input*outer_cell_mask
#
#             outer_cell = outer_forgetgate*outer_cell_previous + outer_ingate*outer_cell_input
#             outer_outgate = slice_outer(outer_gates, 3) + outer_cell*self.W_outer_cell_to_outgate
#
#             if self.grad_clipping:
#                 outer_outgate = theano.gradient.grad_clip(outer_outgate,
#                                                           -self.grad_clipping,
#                                                           self.grad_clipping)
#             if self.use_layer_norm:
#                 outer_outgate = self.layer_norm(input=outer_outgate,
#                                                 alpha=self.alpha_outer_outgate,
#                                                 beta=self.beta_outer_outgate)
#
#             outer_outgate = T.nnet.sigmoid(outer_outgate)
#
#             if self.use_layer_norm:
#                 _cell = self.layer_norm(input=outer_cell,
#                                         alpha=self.alpha_outer_outcell,
#                                         beta=self.beta_outer_outcell)
#             else:
#                 _cell = outer_cell
#
#             if self.grad_clipping:
#                 _cell = theano.gradient.grad_clip(_cell,
#                                                   -self.grad_clipping,
#                                                   self.grad_clipping)
#
#             outer_hid = outer_outgate*T.tanh(_cell)
#             return [inner_cell, inner_hid, outer_cell, outer_hid]
#
#         def step_masked(inner_input_n,
#                         outer_input_n,
#                         mask_input_n,
#                         inner_cell_previous,
#                         inner_hid_previous,
#                         outer_cell_previous,
#                         outer_hid_previous,
#                         *args):
#             inner_cell, inner_hid, outer_cell, outer_hid = step(inner_input_n,
#                                                                 outer_input_n,
#                                                                 inner_cell_previous,
#                                                                 inner_hid_previous,
#                                                                 outer_cell_previous,
#                                                                 outer_hid_previous,
#                                                                 *args)
#
#             inner_cell = T.switch(mask_input_n, inner_cell, inner_cell_previous)
#             inner_hid = T.switch(mask_input_n, inner_hid, inner_hid_previous)
#             outer_cell = T.switch(mask_input_n, outer_cell, outer_cell_previous)
#             outer_hid = T.switch(mask_input_n, outer_hid, outer_hid_previous)
#             return [inner_cell, inner_hid, outer_cell, outer_hid]
#
#         # sequence input
#         sequences = [inner_input,
#                      outer_input,
#                      mask_input]
#         step_fun = step_masked
#
#         # state init
#         ones = T.ones((num_batch, 1))
#         if not isinstance(self.inner_cell_init, Layer):
#             inner_cell_init = T.dot(ones, self.inner_cell_init)
#
#         if not isinstance(self.inner_hid_init, Layer):
#             inner_hid_init = T.dot(ones, self.inner_hid_init)
#
#         if not isinstance(self.outer_cell_init, Layer):
#             outer_cell_init = T.dot(ones, self.outer_cell_init)
#
#         if not isinstance(self.outer_hid_init, Layer):
#             outer_hid_init = T.dot(ones, self.outer_hid_init)
#
#         # non sequence
#         non_seqs = [inner_cell_mask, outer_cell_mask]
#
#         non_seqs += [W_inner_hid_concat,
#                      W_outer_hid_concat,
#                      W_fact_in_concat,
#                      W_fact_hid_concat,
#                      W_fact_b_concat,
#                      b_fact_in_concat,
#                      b_fact_hid_concat]
#
#         non_seqs += [self.W_fact_to_in_ingate,
#                      self.W_fact_to_in_forgetgate,
#                      self.W_fact_to_in_cell,
#                      self.W_fact_to_in_outgate,
#                      self.W_fact_to_hid_ingate,
#                      self.W_fact_to_hid_forgetgate,
#                      self.W_fact_to_hid_cell,
#                      self.W_fact_to_hid_outgate,
#                      self.W_fact_to_b_ingate,
#                      self.W_fact_to_b_forgetgate,
#                      self.W_fact_to_b_cell,
#                      self.W_fact_to_b_outgate,]
#
#         non_seqs += [self.W_inner_cell_to_ingate,
#                      self.W_inner_cell_to_forgetgate,
#                      self.W_inner_cell_to_outgate,
#                      self.W_outer_cell_to_ingate,
#                      self.W_outer_cell_to_forgetgate,
#                      self.W_outer_cell_to_outgate]
#
#         if self.use_layer_norm:
#             non_seqs +=[self.alpha_inner_ingate,
#                         self.alpha_inner_forgetgate,
#                         self.alpha_inner_cell,
#                         self.alpha_inner_outgate,
#                         self.alpha_inner_outcell,
#                         self.beta_inner_ingate,
#                         self.beta_inner_forgetgate,
#                         self.beta_inner_cell,
#                         self.beta_inner_outgate,
#                         self.beta_inner_outcell,]
#
#             non_seqs +=[self.alpha_outer_ingate,
#                         self.alpha_outer_forgetgate,
#                         self.alpha_outer_cell,
#                         self.alpha_outer_outgate,
#                         self.alpha_outer_outcell,
#                         self.beta_outer_ingate,
#                         self.beta_outer_forgetgate,
#                         self.beta_outer_cell,
#                         self.beta_outer_outgate,
#                         self.beta_outer_outcell,]
#         [inner_cell_out,
#          inner_hid_out,
#          outer_cell_out,
#          outer_hid_out] = theano.scan(fn=step_fun,
#                                       sequences=sequences,
#                                       outputs_info=[inner_cell_init,
#                                                     inner_hid_init,
#                                                     outer_cell_init,
#                                                     outer_hid_init],
#                                       go_backwards=self.backwards,
#                                       truncate_gradient=self.gradient_steps,
#                                       non_sequences=non_seqs,
#                                       strict=True)[0]
#
#         inner_hid_out = inner_hid_out.dimshuffle(1, 0, 2)
#         outer_hid_out = outer_hid_out.dimshuffle(1, 0, 2)
#         if self.backwards:
#             inner_hid_out = inner_hid_out[:, ::-1]
#             outer_hid_out = outer_hid_out[:, ::-1]
#
#         if self.only_return_outer:
#             return outer_hid_out
#         else:
#             return T.concatenate([inner_hid_out, outer_hid_out], axis=-1)
