import theano
import numpy
from theano import tensor as T
from lasagne import init, nonlinearities
from lasagne.layers import Layer, MergeLayer, Gate
floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

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
            outer_cell = outer_forget*outer_hid_previous + outer_in*T.tanh(outer_cell)

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

# Based on the LSTMLayer in lasange
# http://lasagne.readthedocs.io/en/latest/modules/layers/recurrent.html#lasagne.layers.LSTMLayer
# Implemented as described in https://arxiv.org/abs/1609.09106.
class HyperLSTMLayer(MergeLayer):
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
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input and the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(HyperLSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
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

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        num_inputs = numpy.prod(input_shape[2:])

        self.nonlinearity_ingate = ingate.nonlinearity
        self.nonlinearity_forgetgate = forgetgate.nonlinearity
        self.nonlinearity_cell = cell.nonlinearity
        self.nonlinearity_outgate = outgate.nonlinearity

        # Equation 10
        def add_hyper_gate_params(gate, gate_name):
            # (W_hhat, W_xhat, bhat)
            return (self.add_param(gate.W_hid, (num_hyper_units, num_hyper_units),
                                   name="W_hhat_{}".format(gate_name)),
                    self.add_param(gate.W_in, (num_inputs+num_units, num_hyper_units),
                                   name="W_xhat_{}".format(gate_name)),
                    self.add_param(gate.b, (num_hyper_units,),
                                   name="bhat_{}".format(gate_name),
                                   regularizable=False))

        # Add in parameters from the supplied Gate instances
        (self.W_hhat_ig, self.W_xhat_ig, self.bhat_ig) = add_hyper_gate_params(ingate, 'ig')
        (self.W_hhat_fg, self.W_xhat_fg, self.bhat_fg) = add_hyper_gate_params(forgetgate, 'fg')
        (self.W_hhat_c, self.W_xhat_c, self.bhat_c) = add_hyper_gate_params(cell, 'c')
        (self.W_hhat_og, self.W_xhat_og, self.bhat_og) = add_hyper_gate_params(outgate, 'og')

        self.hyper_cell_init = self.add_param(
            cell_init, (1, num_hyper_units), name="hyper_cell_init",
            trainable=False, regularizable=False)

        self.hyper_hid_init = self.add_param(
            hid_init, (1, num_hyper_units), name="hyper_hid_init",
            trainable=False, regularizable=False)

        # Equation 11
        def add_proj_params(gate_name):
            """Initalization as described in the paper"""

            # (W_hhat_h, b_hhat, W_hhat_x, b_hhat_x, W_hhat_b)
            return (self.add_param(init.Constant(0.), (num_hyper_units, num_proj_units),
                                   name="W_hhat_h_{}".format(gate_name)),
                    self.add_param(init.Constant(1.), (num_proj_units,),
                                   name="b_hhat_h_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(init.Constant(0.), (num_hyper_units, num_proj_units),
                                   name="W_hhat_x_{}".format(gate_name)),
                    self.add_param(init.Constant(1.), (num_proj_units,),
                                   name="b_hhat_x_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(init.Constant(0.), (num_hyper_units, num_proj_units),
                                   name="W_hhat_b_{}".format(gate_name)))

        (self.W_hhat_h_ig, self.b_hhat_h_ig, self.W_hhat_x_ig, self.b_hhat_x_ig,
         self.W_hhat_b_ig) = add_proj_params('ig')

        (self.W_hhat_h_fg, self.b_hhat_h_fg, self.W_hhat_x_fg, self.b_hhat_x_fg,
         self.W_hhat_b_fg) = add_proj_params('fg')

        (self.W_hhat_h_c, self.b_hhat_h_c, self.W_hhat_x_c, self.b_hhat_x_c,
         self.W_hhat_b_c) = add_proj_params('c')

        (self.W_hhat_h_og, self.b_hhat_h_og, self.W_hhat_x_og, self.b_hhat_x_og,
         self.W_hhat_b_og) = add_proj_params('og')

        # Equation 12
        def add_scale_params(gate_name):
            # (W_hz, W_xz, W_bz, b_0)
            return (self.add_param(init.Constant(1.0/num_proj_units), (num_proj_units, num_units),
                                   name="W_hz_{}".format(gate_name)),
                    self.add_param(init.Constant(1.0/num_proj_units), (num_proj_units, num_units),
                                   name="W_xz_{}".format(gate_name)),
                    self.add_param(init.Constant(0.), (num_proj_units, num_units),
                                   name="W_bz_{}".format(gate_name)),
                    self.add_param(init.Constant(0.), (num_units,),
                                   name="b_0_{}".format(gate_name),
                                   regularizable=False))

        (self.W_hz_ig, self.W_xz_ig, self.W_bz_ig, self.b_0_ig) = add_scale_params('ig')
        (self.W_hz_fg, self.W_xz_fg, self.W_bz_fg, self.b_0_fg) = add_scale_params('fg')
        (self.W_hz_c, self.W_xz_c, self.W_bz_c, self.b_0_c) = add_scale_params('c')
        (self.W_hz_og, self.W_xz_og, self.W_bz_og, self.b_0_og) = add_scale_params('og')

        def add_gate_params(gate, gate_name):
            # (W_h, W_x)
            return (self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_h_{}".format(gate_name)),
                    self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_x_{}".format(gate_name))
                   )

        (self.W_h_ig, self.W_x_ig) = add_gate_params(ingate, 'ig')
        (self.W_h_fg, self.W_x_fg) = add_gate_params(forgetgate, 'fg')
        (self.W_h_c, self.W_x_c) = add_gate_params(cell, 'c')
        (self.W_h_og, self.W_x_og) = add_gate_params(outgate, 'og')

        self.cell_init = self.add_param(
            cell_init, (1, num_units), name="cell_init",
            trainable=False, regularizable=False)

        self.hid_init = self.add_param(
            hid_init, (1, num_units), name="hid_init",
            trainable=False, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hyper_hid_init = None
        hyper_cell_init = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
   
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        
        # input will be overwritten when precompute_input is True.
        # But hyper network needs original input.
        orig_input = input 

        seq_len, num_batch, _ = input.shape

        # Stack weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        
        # Equation 12
        W_h_stacked = T.concatenate(
            [self.W_h_ig, self.W_h_fg, self.W_h_c, self.W_h_og], axis=1)
        W_x_stacked = T.concatenate(
            [self.W_x_ig, self.W_x_fg, self.W_x_c, self.W_x_og], axis=1)

        W_hz_list = [self.W_hz_ig, self.W_hz_fg, self.W_hz_c, self.W_hz_og]
        W_xz_list = [self.W_xz_ig, self.W_xz_fg, self.W_xz_c, self.W_xz_og]

        W_bz_list = [self.W_bz_ig, self.W_bz_fg, self.W_bz_c, self.W_bz_og]
        b_0_list = [self.b_0_ig, self.b_0_fg, self.b_0_c, self.b_0_og]

        # Equation 10
        W_hhat_stacked = T.concatenate(
            [self.W_hhat_ig, self.W_hhat_fg, self.W_hhat_c, self.W_hhat_og], axis=1)
        W_xhat_stacked = T.concatenate(
            [self.W_xhat_ig, self.W_xhat_fg, self.W_xhat_c, self.W_xhat_og], axis=1)
        bhat_stacked = T.concatenate(
            [self.bhat_ig, self.bhat_fg, self.bhat_c, self.bhat_og], axis=0)

        # Equation 11
        W_hhat_h_stacked = T.concatenate(
            [self.W_hhat_h_ig, self.W_hhat_h_fg, self.W_hhat_h_c, self.W_hhat_h_og], axis=1)
        W_hhat_x_stacked = T.concatenate(
            [self.W_hhat_x_ig, self.W_hhat_x_fg, self.W_hhat_x_c, self.W_hhat_x_og], axis=1)
        W_hhat_b_stacked = T.concatenate(
            [self.W_hhat_b_ig, self.W_hhat_b_fg, self.W_hhat_b_c, self.W_hhat_b_og], axis=1)
        b_hhat_h_stacked = T.concatenate(
            [self.b_hhat_h_ig, self.b_hhat_h_fg, self.b_hhat_h_c, self.b_hhat_h_og], axis=0)
        b_hhat_x_stacked = T.concatenate(
            [self.b_hhat_x_ig, self.b_hhat_x_fg, self.b_hhat_x_c, self.b_hhat_x_og], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).

            # Precompute W_x dot x_t in Equation 12
            input = T.dot(input, W_x_stacked)

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            s = x[:, n*self.num_units:(n+1)*self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def hyper_slice(x, n):
            s = x[:, n*self.num_hyper_units:(n+1)*self.num_hyper_units]
            if self.num_hyper_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def proj_slice(x, n):
            s = x[:, n*self.num_proj_units:(n+1)*self.num_proj_units]
            if self.num_proj_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        # orig_input_n is the original input 
        def step(orig_input_n, input_n, hyper_cell_previous, hyper_hid_previous, cell_previous, hid_previous, *args):
            # Equation 10
            hyper_input_n = T.concatenate([hid_previous, orig_input_n], axis=1)

            hyper_input_n = T.dot(hyper_input_n, W_xhat_stacked) + bhat_stacked
            hyper_gates = hyper_input_n + T.dot(hyper_hid_previous, W_hhat_stacked)
            if self.grad_clipping:
                hyper_gates = theano.gradient.grad_clip(
                    hyper_gates, -self.grad_clipping, self.grad_clipping)
            
            hyper_ig, hyper_fg, hyper_cell_input, hyper_og = \
                [hyper_slice(hyper_gates, i) for i in range(4)]
            hyper_ig = self.nonlinearity_ingate(hyper_ig)
            hyper_fg = self.nonlinearity_forgetgate(hyper_fg)
            hyper_cell_input = self.nonlinearity_cell(hyper_cell_input)

            hyper_cell = hyper_fg * hyper_cell_previous + hyper_ig * hyper_cell_input
            hyper_og = self.nonlinearity_outgate(hyper_og)

            hyper_hid = hyper_og * self.nonlinearity(hyper_cell)
            
            # Equation 11
            z_h = T.dot(hyper_hid, W_hhat_h_stacked) + b_hhat_h_stacked
            z_x = T.dot(hyper_hid, W_hhat_x_stacked) + b_hhat_x_stacked
            z_b = T.dot(hyper_hid, W_hhat_b_stacked)

            # Equation 12
            z_h_list = [proj_slice(z_h, i) for i in range(4)]
            z_x_list = [proj_slice(z_x, i) for i in range(4)]
            z_b_list = [proj_slice(z_b, i) for i in range(4)]

            d_h_list = []
            d_x_list = []
            b_list = []
            for z_h_slice, z_x_slice, z_b_slice, W_hz_slice, W_xz_slice, W_bz_slice, b_0_slice in \
                    zip(z_h_list, z_x_list, z_b_list, W_hz_list, W_xz_list, W_bz_list, b_0_list):
                d_h_list.append(T.dot(z_h_slice, W_hz_slice))
                d_x_list.append(T.dot(z_x_slice, W_xz_slice))
                b_list.append(T.dot(z_b_slice, W_bz_slice)+b_0_slice)
            
            d_h_stacked = T.concatenate(d_h_list, axis=1)
            d_x_stacked = T.concatenate(d_x_list, axis=1)
            b_stacked = T.concatenate(b_list, axis=1)
            
            if not self.precompute_input:
                input_n = T.dot(input_n, W_x_stacked)
            
            input_n = input_n * d_x_stacked + b_stacked
            gates = T.dot(hid_previous, W_h_stacked)
            gates = gates * d_h_stacked + input_n
       
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            ingate, forgetgate, cell_input, outgate = \
                [slice_w(gates, i) for i in range(4)]
        
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            cell = forgetgate*cell_previous + ingate*cell_input
            outgate = self.nonlinearity_outgate(outgate)

            hid = outgate*self.nonlinearity(cell)
            return [hyper_cell, hyper_hid, cell, hid]

        def step_masked(orig_input_n, input_n, mask_n, hyper_cell_previous, 
                hyper_hid_previous, cell_previous, hid_previous, *args):
            hyper_cell, hyper_hid, cell, hid = step(orig_input_n, input_n, 
                hyper_cell_previous, hyper_hid_previous, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hyper_cell = T.switch(mask_n, hyper_cell, hyper_cell_previous)
            hyper_hid = T.switch(mask_n, hyper_hid, hyper_hid_previous)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [hyper_cell, hyper_hid, cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [orig_input, input, mask]
            step_fun = step_masked
        else:
            sequences = [orig_input, input]
            step_fun = step

        ones = T.ones((num_batch, 1))
        # Dot against a 1s vector to repeat to shape (num_batch, num_units)
        hyper_cell_init = T.dot(ones, self.hyper_cell_init)
        hyper_hid_init = T.dot(ones, self.hyper_hid_init)
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_xhat_stacked, bhat_stacked, W_hhat_stacked, W_hhat_h_stacked, 
            b_hhat_h_stacked, W_hhat_x_stacked, b_hhat_x_stacked, W_hhat_b_stacked,
            W_h_stacked, W_x_stacked]
        non_seqs.extend(W_hz_list)
        non_seqs.extend(W_xz_list)
        non_seqs.extend(W_bz_list)
        non_seqs.extend(b_0_list)

        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        hyper_cell_out, hyper_hid_out, cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[hyper_cell_init, hyper_hid_init, cell_init, hid_init],
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1]

        return hid_out