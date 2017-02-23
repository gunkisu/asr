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

class ProjectionHyperLSTMLayer(MergeLayer):
    def __init__(self,
                 # input
                 input_data_layer,
                 input_mask_layer,
                 # model size
                 num_factors,
                 num_units,
                 # gradient
                 gradient_steps=-1,
                 grad_clipping=0,
                 # directions
                 backwards=False,
                 **kwargs):

        # input layers
        incomings = [input_data_layer,
                     input_mask_layer]

        # initialize
        super(ProjectionHyperLSTMLayer, self).__init__(incomings, **kwargs)

        self.backwards = backwards

        self.scale_outer_in = None
        self.scale_outer_hid = None
        self.scale_outer_bias = None

        # options
        self.num_units = num_units
        self.num_factors = num_factors
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping

        # input size
        input_shape = self.input_shapes[0]
        num_inputs = input_shape[-1]

        ###############
        # inner level #
        ###############
        def add_inner_gate_params(gate_name, bias_const=0.0):
            return (#### input-to-hidden ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_inner_{}".format(gate_name)),

                    #### hidden-to-hidden ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_factors, num_units),
                                   name="W_inner_to_inner_{}".format(gate_name)),

                    #### outer-to-hidden ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_factors, num_units),
                                   name="W_outer_to_inner_{}".format(gate_name)),

                    #### bias ####
                    self.add_param(spec=init.Constant(bias_const),
                                   shape=(num_units,),
                                   name="b_inner_{}".format(gate_name),
                                   regularizable=False))

        ####ingate####
        (self.W_in_to_inner_ingate,
         self.W_inner_to_inner_ingate,
         self.W_outer_to_inner_ingate,
         self.b_inner_ingate) = add_inner_gate_params('ingate')

        ####forgetgate####
        (self.W_in_to_inner_forgetgate,
         self.W_inner_to_inner_forgetgate,
         self.W_outer_to_inner_forgetgate,
         self.b_inner_forgetgate) = add_inner_gate_params('forgetgate')

        ####cell####
        (self.W_in_to_inner_cell,
         self.W_inner_to_inner_cell,
         self.W_outer_to_inner_cell,
         self.b_inner_cell) = add_inner_gate_params('cell')

        ####outgate####
        (self.W_in_to_inner_outgate,
         self.W_inner_to_inner_outgate,
         self.W_outer_to_inner_outgate,
         self.b_inner_outgate) = add_inner_gate_params('outgate')

        ####projection####
        self.W_inner_prj = self.add_param(spec=init.Orthogonal(),
                                          shape=(num_units, num_factors),
                                          name="W_inner_prj")

        ####cell_ln####
        self.W_inner_cell_ln = self.add_param(spec=init.Constant(1.0),
                                              shape=(self.num_units,),
                                              name="W_inner_cell_ln")
        self.b_inner_cell_ln = self.add_param(spec=init.Constant(0.0),
                                              shape=(self.num_units,),
                                              name="b_inner_cell_ln",
                                              regularizable=False)

        ###############
        # outer level #
        ###############
        def add_outer_gate_params(gate_name, bias_const=0.0):
            return (#### input-to-hidden ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_outer_{}".format(gate_name)),

                    #### hidden-to-hidden ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_factors, num_units),
                                   name="W_outer_to_outer_{}".format(gate_name)),

                    #### scale input-to-hidden ####
                    self.add_param(spec=init.Constant(0.),
                                   shape=(num_factors, num_units),
                                   name="W_inner_to_in_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(1.),
                                   shape=(num_units,),
                                   name="b_outer_to_in_{}".format(gate_name),
                                   regularizable=False),

                    #### scale hidden-to-hidden ####
                    self.add_param(spec=init.Constant(0.),
                                   shape=(num_factors, num_units),
                                   name="W_inner_to_outer_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(1.),
                                   shape=(num_units,),
                                   name="b_inner_to_outer_{}".format(gate_name),
                                   regularizable=False),

                    #### question-to-bias ####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_factors, num_units),
                                   name="W_inner_to_bias_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(bias_const),
                                   shape=(num_units,),
                                   name="b_inner_to_bias_{}".format(gate_name),
                                   regularizable=False))

        ####ingate####
        (self.W_in_to_outer_ingate,
         self.W_outer_to_outer_ingate,
         self.W_inner_to_in_ingate,
         self.b_inner_to_in_ingate,
         self.W_inner_to_outer_ingate,
         self.b_inner_to_outer_ingate,
         self.W_inner_to_bias_ingate,
         self.b_inner_to_bias_ingate) = add_outer_gate_params('ingate')

        ####forgetgate####
        (self.W_in_to_outer_forgetgate,
         self.W_outer_to_outer_forgetgate,
         self.W_inner_to_in_forgetgate,
         self.b_inner_to_in_forgetgate,
         self.W_inner_to_outer_forgetgate,
         self.b_inner_to_outer_forgetgate,
         self.W_inner_to_bias_forgetgate,
         self.b_inner_to_bias_forgetgate) = add_outer_gate_params('forgetgate')

        ####cell####
        (self.W_in_to_outer_cell,
         self.W_outer_to_outer_cell,
         self.W_inner_to_in_cell,
         self.b_inner_to_in_cell,
         self.W_inner_to_outer_cell,
         self.b_inner_to_outer_cell,
         self.W_inner_to_bias_cell,
         self.b_inner_to_bias_cell) = add_outer_gate_params('cell')

        ####outgate####
        (self.W_in_to_outer_outgate,
         self.W_outer_to_outer_outgate,
         self.W_inner_to_in_outgate,
         self.b_inner_to_in_outgate,
         self.W_inner_to_outer_outgate,
         self.b_inner_to_outer_outgate,
         self.W_inner_to_bias_outgate,
         self.b_inner_to_bias_outgate) = add_outer_gate_params('outgate')

        ####projection####
        self.W_outer_prj = self.add_param(spec=init.Orthogonal(),
                                          shape=(num_units, num_factors),
                                          name="W_outer_prj")

        ####cell_ln####
        self.W_outer_cell_ln = self.add_param(spec=init.Constant(1.0),
                                              shape=(self.num_units,),
                                              name="W_outer_cell_ln")
        self.b_outer_cell_ln = self.add_param(spec=init.Constant(0.0),
                                              shape=(self.num_units,),
                                              name="b_outer_cell_ln",
                                              regularizable=False)

        #### initialize ####
        self.inner_cell_init = self.add_param(spec=init.Constant(0.0),
                                              shape=(1, self.num_units),
                                              name="inner_cell_init",
                                              trainable=False,
                                              regularizable=False)
        self.inner_hid_init = self.add_param(spec=init.Constant(0.0),
                                             shape=(1, self.num_factors),
                                             name="inner_hid_init",
                                             trainable=False,
                                             regularizable=False)
        self.outer_cell_init = self.add_param(spec=init.Constant(0.0),
                                              shape=(1, self.num_units),
                                              name="outer_cell_init",
                                              trainable=False,
                                              regularizable=False)
        self.outer_hid_init = self.add_param(spec=init.Constant(0.0),
                                             shape=(1, self.num_factors),
                                             name="outer_hid_init",
                                             trainable=False,
                                             regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        num_outputs = self.num_factors*2

        return input_shape[0], input_shape[1], num_outputs

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # layer inputs
        input_data = inputs[0]
        input_mask = inputs[1]

        # get input size
        input_data = input_data.dimshuffle(1, 0, 2)
        input_mask = input_mask.dimshuffle(1, 0, 'x')
        seq_len, num_batch, num_inputs = input_data.shape


        def slice_block(x, n):
            return x[:, 4*n*self.num_units:4*(n+1)*self.num_units]
        def slice_unit(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        ###############
        # inner level #
        ###############
        W_in_to_inner_concat = T.concatenate([self.W_in_to_inner_ingate,
                                              self.W_in_to_inner_forgetgate,
                                              self.W_in_to_inner_cell,
                                              self.W_in_to_inner_outgate], axis=1)

        W_inner_to_inner_concat = T.concatenate([self.W_inner_to_inner_ingate,
                                                 self.W_inner_to_inner_forgetgate,
                                                 self.W_inner_to_inner_cell,
                                                 self.W_inner_to_inner_outgate], axis=1)

        W_outer_to_inner_concat = T.concatenate([self.W_outer_to_inner_ingate,
                                                 self.W_outer_to_inner_forgetgate,
                                                 self.W_outer_to_inner_cell,
                                                 self.W_outer_to_inner_outgate], axis=1)

        b_inner_concat = T.concatenate([self.b_inner_ingate,
                                        self.b_inner_forgetgate,
                                        self.b_inner_cell,
                                        self.b_inner_outgate], axis=0)

        ##########
        # answer #
        ##########
        W_in_to_outer_concat = T.concatenate([self.W_in_to_outer_ingate,
                                              self.W_in_to_outer_forgetgate,
                                              self.W_in_to_outer_cell,
                                              self.W_in_to_outer_outgate], axis=1)

        W_outer_to_outer_concat = T.concatenate([self.W_outer_to_outer_ingate,
                                                 self.W_outer_to_outer_forgetgate,
                                                 self.W_outer_to_outer_cell,
                                                 self.W_outer_to_outer_outgate], axis=1)

        W_inner_to_outer_concat = T.concatenate([self.W_inner_to_in_ingate,
                                                 self.W_inner_to_in_forgetgate,
                                                 self.W_inner_to_in_cell,
                                                 self.W_inner_to_in_outgate,
                                                 self.W_inner_to_outer_ingate,
                                                 self.W_inner_to_outer_forgetgate,
                                                 self.W_inner_to_outer_cell,
                                                 self.W_inner_to_outer_outgate,
                                                 self.W_inner_to_bias_ingate,
                                                 self.W_inner_to_bias_forgetgate,
                                                 self.W_inner_to_bias_cell,
                                                 self.W_inner_to_bias_outgate], axis=1)

        b_inner_to_outer_concat = T.concatenate([self.b_inner_to_in_ingate,
                                                 self.b_inner_to_in_forgetgate,
                                                 self.b_inner_to_in_cell,
                                                 self.b_inner_to_in_outgate,
                                                 self.b_inner_to_outer_ingate,
                                                 self.b_inner_to_outer_forgetgate,
                                                 self.b_inner_to_outer_cell,
                                                 self.b_inner_to_outer_outgate,
                                                 self.b_inner_to_bias_ingate,
                                                 self.b_inner_to_bias_forgetgate,
                                                 self.b_inner_to_bias_cell,
                                                 self.b_inner_to_bias_outgate], axis=0)

        inner_input = T.dot(input_data, W_in_to_inner_concat) + b_inner_concat
        outer_input = T.dot(input_data, W_in_to_outer_concat)

        def step(inner_input_n,
                 outer_input_n,
                 inner_cell_previous,
                 inner_hid_previous,
                 outer_cell_previous,
                 outer_hid_previous,
                 *args):
            ###############
            # inner level #
            ###############
            inner_gates = inner_input_n
            inner_gates += T.dot(inner_hid_previous, W_inner_to_inner_concat)
            inner_gates += T.dot(outer_hid_previous, W_outer_to_inner_concat)

            inner_in = slice_unit(inner_gates, 0)
            inner_forget = slice_unit(inner_gates, 1)
            inner_cell = slice_unit(inner_gates, 2)
            inner_out = slice_unit(inner_gates, 3)

            if self.grad_clipping:
                inner_in = theano.gradient.grad_clip(inner_in,
                                                     -self.grad_clipping,
                                                     self.grad_clipping)
                inner_forget = theano.gradient.grad_clip(inner_forget,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                inner_cell = theano.gradient.grad_clip(inner_cell,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                inner_out = theano.gradient.grad_clip(inner_out,
                                                      -self.grad_clipping,
                                                      self.grad_clipping)

            inner_in = T.nnet.sigmoid(inner_in)
            inner_forget = T.nnet.sigmoid(inner_forget)
            inner_out = T.nnet.sigmoid(inner_out)
            inner_cell = inner_forget*inner_cell_previous + inner_in*T.tanh(inner_cell)

            inner_outcell = ln(inner_cell, self.W_inner_cell_ln, self.b_inner_cell_ln)
            if self.grad_clipping:
                inner_outcell = theano.gradient.grad_clip(inner_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            inner_hid = inner_out*T.tanh(inner_outcell)
            inner_hid = T.dot(inner_hid, self.W_inner_prj)

            ######### question hidden-to-answer #########
            scale_outer = T.dot(inner_hid, W_inner_to_outer_concat) + b_inner_to_outer_concat
            scale_outer_in = slice_block(scale_outer, 0)
            scale_outer_hid = slice_block(scale_outer, 1)
            scale_outer_bias = slice_block(scale_outer, 2)

            ################
            # outer level #
            ################
            outer_gates = outer_input_n*scale_outer_in
            outer_gates += T.dot(outer_hid_previous, W_outer_to_outer_concat)*scale_outer_hid
            outer_gates += scale_outer_bias

            outer_in = slice_unit(outer_gates, 0)
            outer_forget = slice_unit(outer_gates, 1)
            outer_cell = slice_unit(outer_gates, 2)
            outer_out = slice_unit(outer_gates, 3)

            if self.grad_clipping:
                outer_in = theano.gradient.grad_clip(outer_in,
                                                     -self.grad_clipping,
                                                     self.grad_clipping)
                outer_forget = theano.gradient.grad_clip(outer_forget,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                outer_cell = theano.gradient.grad_clip(outer_cell,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)
                outer_out = theano.gradient.grad_clip(outer_out,
                                                      -self.grad_clipping,
                                                      self.grad_clipping)

            outer_in = T.nnet.sigmoid(outer_in)
            outer_forget = T.nnet.sigmoid(outer_forget)
            outer_out = T.nnet.sigmoid(outer_out)
            outer_cell = outer_forget*outer_cell_previous + outer_in*T.tanh(outer_cell)

            outer_outcell = ln(outer_cell, self.W_outer_cell_ln, self.b_outer_cell_ln)
            if self.grad_clipping:
                outer_outcell = theano.gradient.grad_clip(outer_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            outer_hid = outer_out*T.tanh(outer_outcell)
            outer_hid = T.dot(outer_hid, self.W_outer_prj)
            return [inner_cell,
                    inner_hid,
                    outer_cell,
                    outer_hid,
                    scale_outer_in,
                    scale_outer_hid,
                    scale_outer_bias]

        def step_masked(inner_input_n,
                        outer_input_n,
                        mask_n,
                        inner_cell_previous,
                        inner_hid_previous,
                        outer_cell_previous,
                        outer_hid_previous,
                        *args):
            [inner_cell,
             inner_hid,
             outer_cell,
             outer_hid,
             scale_outer_in,
             scale_outer_hid,
             scale_outer_bias] = step(inner_input_n,
                                      outer_input_n,
                                      inner_cell_previous,
                                      inner_hid_previous,
                                      outer_cell_previous,
                                      outer_hid_previous,
                                      *args)

            inner_cell = T.switch(mask_n, inner_cell, inner_cell_previous)
            inner_hid = T.switch(mask_n, inner_hid, inner_hid_previous)
            outer_cell = T.switch(mask_n, outer_cell, outer_cell_previous)
            outer_hid = T.switch(mask_n, outer_hid, outer_hid_previous)

            scale_outer_in = T.switch(mask_n, scale_outer_in, T.zeros_like(scale_outer_in))
            scale_outer_hid = T.switch(mask_n, scale_outer_hid, T.zeros_like(scale_outer_hid))
            scale_outer_bias = T.switch(mask_n, scale_outer_bias, T.zeros_like(scale_outer_bias))

            return [inner_cell,
                    inner_hid,
                    outer_cell,
                    outer_hid,
                    scale_outer_in,
                    scale_outer_hid,
                    scale_outer_bias]

        sequences = [inner_input,
                     outer_input,
                     input_mask]

        step_fun = step_masked

        ones = T.ones((num_batch, 1))

        inner_cell_init = T.dot(ones, self.inner_cell_init)
        inner_hid_init = T.dot(ones, self.inner_hid_init)
        outer_cell_init = T.dot(ones, self.outer_cell_init)
        outer_hid_init = T.dot(ones, self.outer_hid_init)

        non_seqs = []
        ###############
        # inner level #
        ###############
        non_seqs += [W_inner_to_inner_concat,
                     W_outer_to_inner_concat,
                     self.W_inner_prj]

        ###############
        # outer level #
        ###############
        non_seqs += [W_outer_to_outer_concat,
                     W_inner_to_outer_concat,
                     b_inner_to_outer_concat,
                     self.W_outer_prj]

        non_seqs += [self.W_inner_cell_ln,
                     self.b_inner_cell_ln,
                     self.W_outer_cell_ln,
                     self.b_outer_cell_ln]

        outputs = theano.scan(fn=step_fun,
                              sequences=sequences,
                              outputs_info=[inner_cell_init,
                                            inner_hid_init,
                                            outer_cell_init,
                                            outer_hid_init,
                                            None,
                                            None,
                                            None],
                              truncate_gradient=self.gradient_steps,
                              non_sequences=non_seqs,
                              go_backwards=self.backwards,
                              strict=True)[0]

        inner_out = outputs[-6].dimshuffle(1, 0, 2)
        outer_out = outputs[-4].dimshuffle(1, 0, 2)
        self.scale_outer_in = outputs[-3].dimshuffle(1, 0, 2)
        self.scale_outer_hid = outputs[-2].dimshuffle(1, 0, 2)
        self.scale_outer_bias = outputs[-1].dimshuffle(1, 0, 2)

        if self.backwards:
            inner_out = inner_out[:, ::-1]
            outer_out = outer_out[:, ::-1]
            self.scale_outer_in = self.scale_outer_in[:, ::-1]
            self.scale_outer_hid = self.scale_outer_hid[:, ::-1]
            self.scale_outer_bias = self.scale_outer_bias[:, ::-1]

        return T.concatenate([inner_out, outer_out], axis=-1)


    def get_scale_factors(self, **kwargs):
        return self.scale_outer_in, self.scale_outer_hid, self.scale_outer_bias

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
        num_inner_inputs = inner_input_shape[-1]
        num_outer_inputs = outer_input_shape[-1]

        ##############
        # inner loop #
        ##############
        def add_inner_gate_params(gate_name,
                                  cell_trainable=True,
                                  use_layer_norm=True,
                                  bias_const=0.0):
            return (#### inner input-to-inner (input-to-hidden)####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inner_inputs, num_inner_units),
                                   name="W_inner_in_to_inner_{}".format(gate_name)),

                    #### inner hidden-to-inner (hidden-to-hidden)####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inner_units, num_inner_units),
                                   name="W_inner_hid_to_inner_{}".format(gate_name)),

                    #### inner cell-to-inner (cell-to-hidden)####
                    self.add_param(spec=init.Uniform(0.1) if cell_trainable else init.Constant(0.0),
                                   shape=(num_inner_units,),
                                   name="W_inner_cell_to_inner_{}".format(gate_name),
                                   trainable=cell_trainable),

                    #### outer hidden-to-inner (upper hidden-to-hidden)####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_outer_units, num_inner_units),
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

        ####layer_norm out cell#####
        self.W_inner_ln_outcell = self.add_param(init.Constant(1.),
                                                 shape=(num_inner_units,),
                                                 name="W_inner_ln_outcell",
                                                 trainable=use_layer_norm)
        self.b_inner_ln_outcell =  self.add_param(init.Constant(0.),
                                                  shape=(num_inner_units,),
                                                  name="b_inner_ln_outcell",
                                                  trainable=use_layer_norm,
                                                  regularizable=False)

        ####init_inner_cell####
        self.inner_cell_init = self.add_param(init.Constant(0.),
                                              shape=(1, num_inner_units),
                                              name="inner_cell_init",
                                              trainable=learn_init,
                                              regularizable=False)

        ####init_inner_hidden####
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
                                  use_layer_norm=True,
                                  bias_const=0.0):
            return (#### outer input-to-hidden (input-to-hidden)####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_outer_inputs, num_outer_units),
                                   name="W_outer_in_to_outer_{}".format(gate_name)),
                    #### inner hidden-to-outer in scale ####
                    self.add_param(init.Constant(0.0),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_inner_hid_to_outer_in_{}".format(gate_name)),
                    self.add_param(init.Constant(1.0),
                                   shape=(num_outer_units,),
                                   name="W_inner_hid_to_outer_in_{}".format(gate_name),
                                   regularizable=False),

                    #### outer hidden-to-hidden ####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_outer_units, num_outer_units),
                                   name="W_outer_hid_to_outer_{}".format(gate_name)),
                    #### inner hidden-to-outer hidden scale ####
                    self.add_param(init.Constant(0.0),
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

        ####init_cell####
        self.outer_cell_init = self.add_param(init.Constant(0.),
                                              shape=(1, num_outer_units),
                                              name="outer_cell_init",
                                              trainable=learn_init,
                                              regularizable=False)

        ####init_hid####
        self.outer_hid_init = self.add_param(init.Constant(0.),
                                             shape=(1, num_outer_units),
                                             name="outer_hid_init",
                                             trainable=learn_init,
                                             regularizable=False)

    def layer_norm(self, input, alpha, beta):
        output = (input - T.mean(input, axis=1, keepdims=True))/(T.sqrt(T.var(input, axis=1, keepdims=True) + eps))
        output = alpha[None, :]*output + beta[None, :]
        return output

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_outer:
            num_outputs = self.num_outer_units
        else:
            num_outputs = self.num_inner_units + self.num_outer_units

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
            inner_cell = slice_inner(inner_gates, 2) + inner_cell_previous*self.W_inner_cell_to_inner_cell

            if self.use_layer_norm:
                inner_ingate = self.layer_norm(input=inner_ingate,
                                               alpha=self.W_inner_ln_ingate,
                                               beta=self.b_inner_ln_ingate)
                inner_forgetgate = self.layer_norm(input=inner_forgetgate,
                                                   alpha=self.W_inner_ln_forgetgate,
                                                   beta=self.b_inner_ln_forgetgate)
                inner_cell = self.layer_norm(input=inner_cell,
                                             alpha=self.W_inner_ln_cell,
                                             beta=self.b_inner_ln_cell)

            if self.grad_clipping:
                inner_ingate = theano.gradient.grad_clip(inner_ingate,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                inner_forgetgate = theano.gradient.grad_clip(inner_forgetgate,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
                inner_cell = theano.gradient.grad_clip(inner_cell,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)

            inner_ingate = T.nnet.sigmoid(inner_ingate)
            inner_forgetgate = T.nnet.sigmoid(inner_forgetgate)
            inner_cell = inner_forgetgate*inner_cell_previous + inner_ingate*T.tanh(inner_cell)
            inner_outgate = slice_inner(inner_gates, 3) + inner_cell*self.W_inner_cell_to_inner_outgate
            inner_outcell = inner_cell

            if self.use_layer_norm:
                inner_outgate = self.layer_norm(input=inner_outgate,
                                                alpha=self.W_inner_ln_outgate,
                                                beta=self.b_inner_ln_outgate)
                inner_outcell = self.layer_norm(input=inner_outcell,
                                                alpha=self.W_inner_ln_outcell,
                                                beta=self.b_inner_ln_outcell)
            if self.grad_clipping:
                inner_outgate = theano.gradient.grad_clip(inner_outgate,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)
                inner_outcell = theano.gradient.grad_clip(inner_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            # update inner hidden
            inner_hid = T.nnet.sigmoid(inner_outgate)*T.tanh(inner_outcell)

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
            outer_cell = slice_outer(outer_gates, 2) + outer_cell_previous*slice_outer(scale_outer_cell, 2)

            if self.use_layer_norm:
                outer_ingate = self.layer_norm(input=outer_ingate,
                                               alpha=self.W_outer_ln_ingate,
                                               beta=self.b_outer_ln_ingate)
                outer_forgetgate = self.layer_norm(input=outer_forgetgate,
                                                   alpha=self.W_outer_ln_forgetgate,
                                                   beta=self.b_outer_ln_forgetgate)
                outer_cell = self.layer_norm(input=outer_cell,
                                             alpha=self.W_outer_ln_cell,
                                             beta=self.b_outer_ln_cell)

            if self.grad_clipping:
                outer_ingate = theano.gradient.grad_clip(outer_ingate,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                outer_forgetgate = theano.gradient.grad_clip(outer_forgetgate,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
                outer_cell = theano.gradient.grad_clip(outer_cell,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)

            # get gate nonlinear
            outer_ingate = T.nnet.sigmoid(outer_ingate)
            outer_forgetgate = T.nnet.sigmoid(outer_forgetgate)
            outer_cell = outer_forgetgate*outer_cell_previous + outer_ingate*T.tanh(outer_cell)
            outer_outgate = slice_outer(outer_gates, 3) + outer_cell*slice_outer(scale_outer_cell, 3)
            outer_outcell = outer_cell

            if self.use_layer_norm:
                outer_outgate = self.layer_norm(input=outer_outgate,
                                                alpha=self.W_outer_ln_outgate,
                                                beta=self.b_outer_ln_outgate)
                outer_outcell = self.layer_norm(input=outer_outcell,
                                                alpha=self.W_outer_ln_outcell,
                                                beta=self.b_outer_ln_outcell)

            if self.grad_clipping:
                outer_outgate = theano.gradient.grad_clip(outer_outgate,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)
                outer_outcell = theano.gradient.grad_clip(outer_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            outer_hid = T.nnet.sigmoid(outer_outgate)*T.tanh(outer_outcell)
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

        ##############
        # inner loop #
        ##############
        non_seqs = [W_inner_hid_to_inner_concat,
                    W_inner_cell_to_inner_concat,
                    W_outer_hid_to_inner_concat]

        non_seqs += [self.W_inner_cell_to_inner_ingate,
                     self.W_inner_cell_to_inner_forgetgate,
                     self.W_inner_cell_to_inner_cell,
                     self.W_inner_cell_to_inner_outgate]

        ##############
        # outer loop #
        ##############
        non_seqs += [W_outer_hid_to_outer_concat,
                     W_inner_hid_to_outer_in_concat,
                     b_inner_hid_to_outer_in_concat,
                     W_inner_hid_to_outer_hid_concat,
                     b_inner_hid_to_outer_hid_concat,
                     W_inner_hid_to_outer_cell_concat,
                     b_inner_hid_to_outer_cell_concat,
                     W_inner_hid_to_outer_bias_concat,
                     b_inner_hid_to_outer_bias_concat]

        # layer norm
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
                        self.b_inner_ln_outcell]

            non_seqs +=[self.W_outer_ln_ingate,
                        self.W_outer_ln_forgetgate,
                        self.W_outer_ln_cell,
                        self.W_outer_ln_outgate,
                        self.W_outer_ln_outcell,
                        self.b_outer_ln_ingate,
                        self.b_outer_ln_forgetgate,
                        self.b_outer_ln_cell,
                        self.b_outer_ln_outgate,
                        self.b_outer_ln_outcell]

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

class EqualScaleHyperLSTMLayer(MergeLayer):
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
        super(EqualScaleHyperLSTMLayer, self).__init__(incomings, **kwargs)

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
        num_inner_inputs = inner_input_shape[-1]
        num_outer_inputs = outer_input_shape[-1]

        ##############
        # inner loop #
        ##############
        def add_inner_gate_params(gate_name,
                                  cell_trainable=True,
                                  use_layer_norm=True,
                                  bias_const=0.0):
            return (#### inner input-to-inner (input-to-hidden)####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inner_inputs, num_outer_units),
                                   name="W_inner_in_to_inner_{}".format(gate_name)),

                    #### inner hidden-to-inner (hidden-to-hidden)####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_inner_hid_to_inner_{}".format(gate_name)),

                    #### inner cell-to-inner (cell-to-hidden)####
                    self.add_param(spec=init.Uniform(0.1) if cell_trainable else init.Constant(0.0),
                                   shape=(num_outer_units,),
                                   name="W_inner_cell_to_inner_{}".format(gate_name),
                                   trainable=cell_trainable),

                    #### outer hidden-to-inner (upper hidden-to-hidden)####
                    self.add_param(spec=init.Orthogonal(),
                                   shape=(num_outer_units, num_outer_units),
                                   name="W_outer_hid_to_inner_{}".format(gate_name)),

                    #### bias ####
                    self.add_param(spec=init.Constant(bias_const),
                                   shape=(num_outer_units,),
                                   name="b_inner_{}".format(gate_name),
                                   regularizable=False),

                    #### layer norm ####
                    self.add_param(spec=init.Constant(1.),
                                   shape=(num_outer_units,),
                                   name="W_inner_ln_{}".format(gate_name),
                                   trainable=use_layer_norm),
                    self.add_param(spec=init.Constant(0.),
                                   shape=(num_outer_units,),
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

        #### hidden projection #####
        self.W_inner_hid_prj = self.add_param(init.Orthogonal(),
                                              shape=(num_outer_units, num_inner_units),
                                              name="W_inner_hid_prj")

        ####layer_norm out cell#####
        self.W_inner_ln_outcell = self.add_param(init.Constant(1.),
                                                 shape=(num_outer_units,),
                                                 name="W_inner_ln_outcell",
                                                 trainable=use_layer_norm)
        self.b_inner_ln_outcell =  self.add_param(init.Constant(0.),
                                                  shape=(num_outer_units,),
                                                  name="b_inner_ln_outcell",
                                                  trainable=use_layer_norm,
                                                  regularizable=False)

        ####init_inner_cell####
        self.inner_cell_init = self.add_param(init.Constant(0.),
                                              shape=(1, num_outer_units),
                                              name="inner_cell_init",
                                              trainable=learn_init,
                                              regularizable=False)

        ####init_inner_hidden####
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
                                  use_layer_norm=True,
                                  bias_const=0.0):
            return (#### outer input-to-hidden (input-to-hidden)####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_outer_inputs, num_outer_units),
                                   name="W_outer_in_to_outer_{}".format(gate_name)),
                    #### inner hidden-to-outer in scale ####
                    self.add_param(init.Constant(0.0),
                                   shape=(num_inner_units, num_outer_units),
                                   name="W_inner_hid_to_outer_in_{}".format(gate_name)),
                    self.add_param(init.Constant(1.0),
                                   shape=(num_outer_units,),
                                   name="W_inner_hid_to_outer_in_{}".format(gate_name),
                                   regularizable=False),

                    #### outer hidden-to-hidden ####
                    self.add_param(init.Orthogonal(),
                                   shape=(num_outer_units, num_outer_units),
                                   name="W_outer_hid_to_outer_{}".format(gate_name)),
                    #### inner hidden-to-outer hidden scale ####
                    self.add_param(init.Constant(0.0),
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

        ####init_cell####
        self.outer_cell_init = self.add_param(init.Constant(0.),
                                              shape=(1, num_outer_units),
                                              name="outer_cell_init",
                                              trainable=learn_init,
                                              regularizable=False)

        ####init_hid####
        self.outer_hid_init = self.add_param(init.Constant(0.),
                                             shape=(1, num_outer_units),
                                             name="outer_hid_init",
                                             trainable=learn_init,
                                             regularizable=False)

    def layer_norm(self, input, alpha, beta):
        output = (input - T.mean(input, axis=1, keepdims=True))/(T.sqrt(T.var(input, axis=1, keepdims=True) + eps))
        output = alpha[None, :]*output + beta[None, :]
        return output

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_outer:
            num_outputs = self.num_outer_units
        else:
            num_outputs = self.num_inner_units + self.num_outer_units

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

            inner_ingate = slice_outer(inner_gates, 0) + inner_cell_previous*self.W_inner_cell_to_inner_ingate
            inner_forgetgate = slice_outer(inner_gates, 1) + inner_cell_previous*self.W_inner_cell_to_inner_forgetgate
            inner_cell = slice_outer(inner_gates, 2) + inner_cell_previous*self.W_inner_cell_to_inner_cell

            if self.use_layer_norm:
                inner_ingate = self.layer_norm(input=inner_ingate,
                                               alpha=self.W_inner_ln_ingate,
                                               beta=self.b_inner_ln_ingate)
                inner_forgetgate = self.layer_norm(input=inner_forgetgate,
                                                   alpha=self.W_inner_ln_forgetgate,
                                                   beta=self.b_inner_ln_forgetgate)
                inner_cell = self.layer_norm(input=inner_cell,
                                             alpha=self.W_inner_ln_cell,
                                             beta=self.b_inner_ln_cell)

            if self.grad_clipping:
                inner_ingate = theano.gradient.grad_clip(inner_ingate,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                inner_forgetgate = theano.gradient.grad_clip(inner_forgetgate,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
                inner_cell = theano.gradient.grad_clip(inner_cell,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)

            inner_ingate = T.nnet.sigmoid(inner_ingate)
            inner_forgetgate = T.nnet.sigmoid(inner_forgetgate)
            inner_cell = inner_forgetgate*inner_cell_previous + inner_ingate*T.tanh(inner_cell)
            inner_outgate = slice_outer(inner_gates, 3) + inner_cell*self.W_inner_cell_to_inner_outgate
            inner_outcell = inner_cell

            if self.use_layer_norm:
                inner_outgate = self.layer_norm(input=inner_outgate,
                                                alpha=self.W_inner_ln_outgate,
                                                beta=self.b_inner_ln_outgate)
                inner_outcell = self.layer_norm(input=inner_outcell,
                                                alpha=self.W_inner_ln_outcell,
                                                beta=self.b_inner_ln_outcell)
            if self.grad_clipping:
                inner_outgate = theano.gradient.grad_clip(inner_outgate,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)
                inner_outcell = theano.gradient.grad_clip(inner_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            # update inner hidden
            inner_hid = T.nnet.sigmoid(inner_outgate)*T.tanh(inner_outcell)
            inner_hid = T.dot(inner_hid, self.W_inner_hid_prj)

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
            outer_cell = slice_outer(outer_gates, 2) + outer_cell_previous*slice_outer(scale_outer_cell, 2)

            if self.use_layer_norm:
                outer_ingate = self.layer_norm(input=outer_ingate,
                                               alpha=self.W_outer_ln_ingate,
                                               beta=self.b_outer_ln_ingate)
                outer_forgetgate = self.layer_norm(input=outer_forgetgate,
                                                   alpha=self.W_outer_ln_forgetgate,
                                                   beta=self.b_outer_ln_forgetgate)
                outer_cell = self.layer_norm(input=outer_cell,
                                             alpha=self.W_outer_ln_cell,
                                             beta=self.b_outer_ln_cell)

            if self.grad_clipping:
                outer_ingate = theano.gradient.grad_clip(outer_ingate,
                                                         -self.grad_clipping,
                                                         self.grad_clipping)
                outer_forgetgate = theano.gradient.grad_clip(outer_forgetgate,
                                                             -self.grad_clipping,
                                                             self.grad_clipping)
                outer_cell = theano.gradient.grad_clip(outer_cell,
                                                       -self.grad_clipping,
                                                       self.grad_clipping)

            # get gate nonlinear
            outer_ingate = T.nnet.sigmoid(outer_ingate)
            outer_forgetgate = T.nnet.sigmoid(outer_forgetgate)
            outer_cell = outer_forgetgate*outer_cell_previous + outer_ingate*T.tanh(outer_cell)
            outer_outgate = slice_outer(outer_gates, 3) + outer_cell*slice_outer(scale_outer_cell, 3)
            outer_outcell = outer_cell

            if self.use_layer_norm:
                outer_outgate = self.layer_norm(input=outer_outgate,
                                                alpha=self.W_outer_ln_outgate,
                                                beta=self.b_outer_ln_outgate)
                outer_outcell = self.layer_norm(input=outer_outcell,
                                                alpha=self.W_outer_ln_outcell,
                                                beta=self.b_outer_ln_outcell)

            if self.grad_clipping:
                outer_outgate = theano.gradient.grad_clip(outer_outgate,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)
                outer_outcell = theano.gradient.grad_clip(outer_outcell,
                                                          -self.grad_clipping,
                                                          self.grad_clipping)

            outer_hid = T.nnet.sigmoid(outer_outgate)*T.tanh(outer_outcell)
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

        ##############
        # inner loop #
        ##############
        non_seqs = [W_inner_hid_to_inner_concat,
                    W_inner_cell_to_inner_concat,
                    W_outer_hid_to_inner_concat,
                    self.W_inner_hid_prj]

        non_seqs += [self.W_inner_cell_to_inner_ingate,
                     self.W_inner_cell_to_inner_forgetgate,
                     self.W_inner_cell_to_inner_cell,
                     self.W_inner_cell_to_inner_outgate]

        ##############
        # outer loop #
        ##############
        non_seqs += [W_outer_hid_to_outer_concat,
                     W_inner_hid_to_outer_in_concat,
                     b_inner_hid_to_outer_in_concat,
                     W_inner_hid_to_outer_hid_concat,
                     b_inner_hid_to_outer_hid_concat,
                     W_inner_hid_to_outer_cell_concat,
                     b_inner_hid_to_outer_cell_concat,
                     W_inner_hid_to_outer_bias_concat,
                     b_inner_hid_to_outer_bias_concat]

        # layer norm
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
                        self.b_inner_ln_outcell]

            non_seqs +=[self.W_outer_ln_ingate,
                        self.W_outer_ln_forgetgate,
                        self.W_outer_ln_cell,
                        self.W_outer_ln_outgate,
                        self.W_outer_ln_outcell,
                        self.b_outer_ln_ingate,
                        self.b_outer_ln_forgetgate,
                        self.b_outer_ln_cell,
                        self.b_outer_ln_outgate,
                        self.b_outer_ln_outcell]

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
