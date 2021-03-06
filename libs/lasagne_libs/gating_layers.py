import numpy as np
import theano
import theano.tensor as T
from lasagne import init
from lasagne.layers import Layer, MergeLayer
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX

class SkipLSTMLayer(MergeLayer):
    def __init__(self,
                 # input data
                 input_data_layer,
                 input_mask_layer,
                 # model size
                 num_units,
                 # initialize
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 learn_init=False,
                 # options
                 skip_scale=T.ones(shape=(1,), dtype=floatX),
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 **kwargs):

        # input
        incomings = [input_data_layer,
                     input_mask_layer]

        # init hidden
        self.hid_init_incoming_index = -1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        # init cell
        self.cell_init_incoming_index = -1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # init class
        super(SkipLSTMLayer, self).__init__(incomings, **kwargs)

        # set options
        self.skip_scale = skip_scale
        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final

        # set sampler
        self.uniform = RandomStreams(get_rng().randint(1, 2147462579)).uniform

        # get input size
        input_shape = self.input_shapes[0]
        num_inputs = np.prod(input_shape[2:])

        ###################
        # gate parameters #
        ###################
        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(0.1),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(0.1),
                                   shape=(num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(0.0),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        ##### in gate #####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')
        self.W_cell_to_ingate = self.add_param(spec=init.Uniform(0.1),
                                               shape=(num_units,),
                                               name="W_cell_to_ingate")
        ##### forget gate #####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate')
        self.W_cell_to_forgetgate = self.add_param(spec=init.Uniform(0.1),
                                                   shape=(num_units,),
                                                   name="W_cell_to_forgetgate")
        ##### cell #####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        ##### out gate #####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')
        self.W_cell_to_outgate = self.add_param(spec=init.Uniform(0.1),
                                                shape=(num_units,),
                                                name="W_cell_to_outgate")


        ###################
        # skip parameters #
        ###################
        self.W_cell_to_skip = self.add_param(spec=init.Orthogonal(0.1),
                                             shape=(num_units, num_units),
                                             name="W_cell_to_skip")
        self.b_cell_to_skip = self.add_param(spec=init.Constant(1.0),
                                             shape=(num_units,),
                                             name="b_cell_to_skip",
                                             regularizable=False)

        self.W_hid_to_skip = self.add_param(spec=init.Orthogonal(0.1),
                                            shape=(num_units, num_units),
                                            name="W_hid_to_skip")
        self.b_hid_to_skip = self.add_param(spec=init.Constant(1.0),
                                            shape=(num_units,),
                                            name="b_hid_to_skip",
                                            regularizable=False)

        self.W_in_to_skip = self.add_param(spec=init.Orthogonal(0.1),
                                           shape=(num_inputs, num_units),
                                           name="W_in_to_skip")
        self.b_in_to_skip = self.add_param(spec=init.Constant(0.0),
                                           shape=(num_units,),
                                           name="b_in_to_skip",
                                           regularizable=False)

        self.W_skip = self.add_param(spec=init.Constant(0.0),
                                     shape=(num_units, 1),
                                     name="W_skip")
        self.b_skip = self.add_param(spec=init.Constant(-3.0),
                                     shape=(1,),
                                     name="b_skip",
                                     regularizable=False)

        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(spec=cell_init,
                                            shape=(1, num_units),
                                            name="cell_init",
                                            trainable=learn_init,
                                            regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(spec=hid_init,
                                           shape=(1, num_units),
                                           name="hid_init",
                                           trainable=learn_init,
                                           regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]

        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input_data = inputs[0]
        input_mask = inputs[1]

        hid_init = None
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        cell_init = None
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        if input_data.ndim > 3:
            input_data = T.flatten(input_data, 3)


        input_data = input_data.dimshuffle(1, 0, 2)
        input_mask = input_mask.dimshuffle(1, 0, 'x')

        seq_len, num_batch, num_inputs = input_data.shape

        stochastic_samples = self.uniform(size=(seq_len, 1), dtype=floatX)

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate],
                                     axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate],
                                      axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate],
                                  axis=0)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_data_n,
                 sample_data_n,
                 cell_previous,
                 hid_previous,
                 *args):
            ####################
            # skip computation #
            ####################
            skip_comp = (T.dot(input_data_n, self.W_in_to_skip) + self.b_in_to_skip)
            skip_comp *=(T.dot(cell_previous, self.W_cell_to_skip) + self.b_cell_to_skip)
            skip_comp *=(T.dot(hid_previous, self.W_hid_to_skip) + self.b_hid_to_skip)
            if self.grad_clipping:
                skip_comp = theano.gradient.grad_clip(skip_comp,
                                                      -self.grad_clipping,
                                                      self.grad_clipping)
            skip_comp = T.tanh(skip_comp)
            skip_comp = T.dot(skip_comp,  self.W_skip)
            skip_comp = T.nnet.sigmoid(self.skip_scale*skip_comp + self.b_skip)

            if deterministic:
                skip_comp = T.round(skip_comp)
            else:
                epsilon = theano.gradient.disconnected_grad(T.lt(sample_data_n, skip_comp) - skip_comp)
                skip_comp = skip_comp + epsilon

            ####################
            # lstm computation #
            ####################
            gates = T.dot(input_data_n, W_in_stacked)
            gates += T.dot(hid_previous, W_hid_stacked)
            gates += b_stacked

            ingate = slice_w(gates, 0) + cell_previous*self.W_cell_to_ingate
            forgetgate = slice_w(gates, 1) + cell_previous*self.W_cell_to_forgetgate
            cell_input = slice_w(gates, 2)

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

            outgate = slice_w(gates, 3) + cell*self.W_cell_to_outgate
            if self.grad_clipping:
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(cell)

            skip_comp = T.flatten(skip_comp)

            cell = T.switch(skip_comp[:, None], cell_previous, cell)
            hid = T.switch(skip_comp[:, None], hid_previous, hid)

            return [cell, hid, skip_comp]

        def step_masked(input_data_n,
                        input_mask_n,
                        sample_data_n,
                        cell_previous,
                        hid_previous,
                        *args):
            cell, hid, skip_comp = step(input_data_n,
                                        sample_data_n,
                                        cell_previous,
                                        hid_previous,
                                        *args)

            cell = T.switch(input_mask_n, cell, cell_previous)
            hid = T.switch(input_mask_n, hid, hid_previous)
            return [cell, hid, skip_comp]

        sequences = [input_data,
                     input_mask,
                     stochastic_samples]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_in_stacked,
                    W_hid_stacked,
                    b_stacked]

        non_seqs += [self.W_cell_to_ingate,
                     self.W_cell_to_forgetgate,
                     self.W_cell_to_outgate]

        non_seqs += [self.W_in_to_skip,
                     self.b_in_to_skip,
                     self.W_hid_to_skip,
                     self.b_hid_to_skip,
                     self.W_cell_to_skip,
                     self.b_cell_to_skip,
                     self.W_skip,
                     self.b_skip]

        non_seqs += [self.skip_scale]

        cell_out, hid_out, skip_comp = theano.scan(fn=step_fun,
                                                   sequences=sequences,
                                                   outputs_info=[cell_init, hid_init, None],
                                                   go_backwards=self.backwards,
                                                   truncate_gradient=self.gradient_steps,
                                                   non_sequences=non_seqs,
                                                   n_steps=seq_len,
                                                   strict=True)[0]

        self.skip_comp = skip_comp.dimshuffle(1, 0)

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out

class DiffSkipLSTMLayer(MergeLayer):
    def __init__(self,
                 # input data
                 input_data_layer,
                 input_mask_layer,
                 # model size
                 num_units,
                 # initialize
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 learn_init=False,
                 # options
                 stochastic=False,
                 skip_scale=T.ones(shape=(1,), dtype=floatX),
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 only_return_final=False,
                 **kwargs):

        # input
        incomings = [input_data_layer,
                     input_mask_layer]

        # init input
        input_init = init.Constant(0.)
        self.input_init_incoming_index = -1
        if isinstance(input_init, Layer):
            incomings.append(input_init)
            self.input_init_incoming_index = len(incomings)-1

        # init hidden
        self.hid_init_incoming_index = -1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        # init cell
        self.cell_init_incoming_index = -1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # init class
        super(DiffSkipLSTMLayer, self).__init__(incomings, **kwargs)

        # set options
        self.stochastic = stochastic
        self.skip_scale = skip_scale
        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final

        # set sampler
        self.uniform = RandomStreams(get_rng().randint(1, 2147462579)).uniform

        # get input size
        input_shape = self.input_shapes[0]
        num_inputs = np.prod(input_shape[2:])

        ###################
        # gate parameters #
        ###################
        def add_gate_params(gate_name):
            return (self.add_param(spec=init.Orthogonal(0.1),
                                   shape=(num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(spec=init.Orthogonal(0.1),
                                   shape=(num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(spec=init.Constant(0.0),
                                   shape=(num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False))

        ##### in gate #####
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate) = add_gate_params('ingate')
        self.W_cell_to_ingate = self.add_param(spec=init.Uniform(0.1),
                                               shape=(num_units,),
                                               name="W_cell_to_ingate")
        ##### forget gate #####
        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate) = add_gate_params('forgetgate')
        self.W_cell_to_forgetgate = self.add_param(spec=init.Uniform(0.1),
                                                   shape=(num_units,),
                                                   name="W_cell_to_forgetgate")
        ##### cell #####
        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell) = add_gate_params('cell')

        ##### out gate #####
        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate) = add_gate_params('outgate')
        self.W_cell_to_outgate = self.add_param(spec=init.Uniform(0.1),
                                                shape=(num_units,),
                                                name="W_cell_to_outgate")


        ###################
        # skip parameters #
        ###################
        self.W_cell_to_skip = self.add_param(spec=init.Orthogonal(0.1),
                                             shape=(num_units, num_units),
                                             name="W_cell_to_skip")
        self.b_cell_to_skip = self.add_param(spec=init.Constant(1.0),
                                             shape=(num_units,),
                                             name="b_cell_to_skip",
                                             regularizable=False)

        self.W_hid_to_skip = self.add_param(spec=init.Orthogonal(0.1),
                                            shape=(num_units, num_units),
                                            name="W_hid_to_skip")
        self.b_hid_to_skip = self.add_param(spec=init.Constant(1.0),
                                            shape=(num_units,),
                                            name="b_hid_to_skip",
                                            regularizable=False)

        self.W_in_to_skip = self.add_param(spec=init.Orthogonal(0.1),
                                           shape=(num_inputs, num_units),
                                           name="W_in_to_skip")
        self.b_in_to_skip = self.add_param(spec=init.Constant(1.0),
                                           shape=(num_units,),
                                           name="b_in_to_skip",
                                           regularizable=False)

        self.W_skip = self.add_param(spec=init.Orthogonal(0.1),
                                     shape=(num_units, 1),
                                     name="W_skip")
        self.b_skip = self.add_param(spec=init.Constant(0.0),
                                     shape=(1,),
                                     name="b_skip",
                                     regularizable=False)

        self.W_diff_to_skip = self.add_param(spec=init.Orthogonal(0.1),
                                             shape=(num_inputs, num_units),
                                             name="W_diff_to_skip")
        self.b_diff_to_skip = self.add_param(spec=init.Constant(0.0),
                                             shape=(num_units,),
                                             name="b_diff_to_skip",
                                             regularizable=False)





        if isinstance(input_init, Layer):
            self.input_init = input_init
        else:
            self.input_init = self.add_param(spec=input_init,
                                             shape=(1, num_inputs),
                                             name="input_init",
                                             trainable=learn_init,
                                             regularizable=False)

        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(spec=cell_init,
                                            shape=(1, num_units),
                                            name="cell_init",
                                            trainable=learn_init,
                                            regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(spec=hid_init,
                                           shape=(1, num_units),
                                           name="hid_init",
                                           trainable=learn_init,
                                           regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]

        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input_data = inputs[0]
        input_mask = inputs[1]

        input_init = None
        if self.input_init_incoming_index > 0:
            input_init = inputs[self.input_init_incoming_index]
        hid_init = None
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        cell_init = None
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        if input_data.ndim > 3:
            input_data = T.flatten(input_data, 3)


        input_data = input_data.dimshuffle(1, 0, 2)
        input_mask = input_mask.dimshuffle(1, 0, 'x')

        seq_len, num_batch, num_inputs = input_data.shape

        stochastic_samples = self.uniform(size=(seq_len, 1), dtype=floatX)

        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate],
                                     axis=1)

        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate],
                                      axis=1)

        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate],
                                  axis=0)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_data_n,
                 sample_data_n,
                 input_previous,
                 cell_previous,
                 hid_previous,
                 *args):
            ####################
            # input difference #
            ####################
            input_diff = input_data_n - input_previous
            input = input_data_n

            ####################
            # skip computation #
            ####################
            # if skip_comp is near 1, it skips current time step,
            # otherwise, it works as normal lstm
            # if input_diff is totally 0, skip_comp_scale will be nearly 0
            skip_comp_scale = T.dot(input_diff, self.W_diff_to_skip) + self.b_diff_to_skip
            skip_comp = (T.dot(input_data_n, self.W_in_to_skip) + self.b_in_to_skip)
            skip_comp *=(T.dot(cell_previous, self.W_cell_to_skip) + self.b_cell_to_skip)
            skip_comp *=(T.dot(hid_previous, self.W_hid_to_skip) + self.b_hid_to_skip)


            if self.grad_clipping:
                skip_comp = theano.gradient.grad_clip(skip_comp,
                                                      -self.grad_clipping,
                                                      self.grad_clipping)
                skip_comp_scale = theano.gradient.grad_clip(skip_comp_scale,
                                                            -self.grad_clipping,
                                                            self.grad_clipping)
            skip_comp = T.tanh(skip_comp)
            skip_comp_scale = T.tanh(skip_comp_scale) # if near zero => zero
            skip_comp = T.dot(skip_comp_scale*skip_comp,  self.W_skip)
            skip_comp = T.nnet.sigmoid(self.skip_scale*skip_comp + self.b_skip) # b_skip less than 0

            if deterministic:
                skip_comp = T.round(skip_comp)
            elif self.stochastic:
                epsilon = theano.gradient.disconnected_grad(T.lt(sample_data_n, skip_comp) - skip_comp)
                skip_comp = skip_comp + epsilon
            else:
                epsilon = theano.gradient.disconnected_grad(T.lt(0.5, skip_comp) - skip_comp)
                skip_comp = skip_comp + epsilon


            ####################
            # lstm computation #
            ####################
            gates = T.dot(input_data_n, W_in_stacked)
            gates += T.dot(hid_previous, W_hid_stacked)
            gates += b_stacked

            ingate = slice_w(gates, 0) + cell_previous*self.W_cell_to_ingate
            forgetgate = slice_w(gates, 1) + cell_previous*self.W_cell_to_forgetgate
            cell_input = slice_w(gates, 2)

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

            outgate = slice_w(gates, 3) + cell*self.W_cell_to_outgate
            if self.grad_clipping:
                outgate = theano.gradient.grad_clip(outgate,
                                                    -self.grad_clipping,
                                                    self.grad_clipping)

            outgate = T.nnet.sigmoid(outgate)
            hid = outgate*T.tanh(cell)
            skip_comp = T.flatten(skip_comp)

            cell = T.switch(skip_comp[:, None], cell_previous, cell)
            hid = T.switch(skip_comp[:, None], hid_previous, hid)
            return [input, cell, hid, skip_comp]

        def step_masked(input_data_n,
                        input_mask_n,
                        sample_data_n,
                        input_previous,
                        cell_previous,
                        hid_previous,
                        *args):
            input, cell, hid, skip_comp = step(input_data_n,
                                               sample_data_n,
                                               input_previous,
                                               cell_previous,
                                               hid_previous,
                                               *args)

            input = T.switch(input_mask_n, input, input_previous)
            cell = T.switch(input_mask_n, cell, cell_previous)
            hid = T.switch(input_mask_n, hid, hid_previous)
            return [input, cell, hid, skip_comp]

        sequences = [input_data,
                     input_mask,
                     stochastic_samples]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        if not isinstance(self.input_init, Layer):
            input_init = T.dot(ones, self.input_init)
        if not isinstance(self.cell_init, Layer):
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_in_stacked,
                    W_hid_stacked,
                    b_stacked]

        non_seqs += [self.W_cell_to_ingate,
                     self.W_cell_to_forgetgate,
                     self.W_cell_to_outgate]

        non_seqs += [self.W_in_to_skip,
                     self.b_in_to_skip,
                     self.W_hid_to_skip,
                     self.b_hid_to_skip,
                     self.W_cell_to_skip,
                     self.b_cell_to_skip,
                     self.W_skip,
                     self.b_skip,
                     self.W_diff_to_skip,
                     self.b_diff_to_skip]

        non_seqs += [self.skip_scale]

        [input_out,
         cell_out,
         hid_out,
         skip_comp] = theano.scan(fn=step_fun,
                                  sequences=sequences,
                                  outputs_info=[input_init,
                                                cell_init,
                                                hid_init,
                                                None],
                                  go_backwards=self.backwards,
                                  truncate_gradient=self.gradient_steps,
                                  non_sequences=non_seqs,
                                  n_steps=seq_len,
                                  strict=True)[0]

        self.skip_comp = skip_comp.dimshuffle(1, 0)

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out