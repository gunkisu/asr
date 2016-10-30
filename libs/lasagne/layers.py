import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.layers import Layer, MergeLayer, Gate, ConcatLayer
from lasagne import nonlinearities, init
from lasagne.layers import get_output
from lasagne.utils import unroll_scan
from lasagne.random import get_rng

floatX = theano.config.floatX
eps = np.finfo(floatX).eps

class SequenceSoftmaxLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_outputs,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.),
                 mask_input=None,
                 **kwargs):

        incomings = [incoming,]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1

        self.num_outputs = num_outputs

        super(SequenceSoftmaxLayer, self).__init__(incomings, **kwargs)

        num_inputs = self.input_shapes[0][-1]

        self.W = self.add_param(W, (num_inputs, num_outputs), name="W")

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_outputs,), name="b", regularizable=False)

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

        # softmax operation for probability
        activation = T.exp(activation)
        if mask:
            activation = activation*mask[:, :, None]
        output = activation/(T.sum(activation, axis=-1, keepdims=True) + eps)

        return output



class LSTMLayer(MergeLayer):
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 dropout_ratio=0.2,
                 use_layer_norm=True,
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
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incomings, **kwargs)

        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = dropout_ratio

        self.use_layer_norm = use_layer_norm

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate,
                                                     'ingate')

        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell,
                                                   'cell')

        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate,
                                                      'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        if self.use_layer_norm:
            self.alpha_gate = self.add_param(init.Constant(1.),
                                             (num_units*4,),
                                             name="alpha_gate")
            self.beta_gate =  self.add_param(init.Constant(0.),
                                             (num_units*4,),
                                             name="beta_gate",
                                             regularizable=False)

            self.alpha_cell = self.add_param(init.Constant(1.),
                                             (num_units,),
                                             name="alpha_cell")
            self.beta_cell =  self.add_param(init.Constant(0.),
                                             (num_units,),
                                             name="beta_cell",
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
        output = (input - T.mean(input, axis=1, keepdims=True))/T.sqrt(T.var(input, axis=1, keepdims=True) + eps)
        output = alpha[None, :]*output + beta[None, :]
        return output

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
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
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate([self.W_in_to_ingate,
                                      self.W_in_to_forgetgate,
                                      self.W_in_to_cell,
                                      self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate([self.W_hid_to_ingate,
                                       self.W_hid_to_forgetgate,
                                       self.W_hid_to_cell,
                                       self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate([self.b_ingate,
                                   self.b_forgetgate,
                                   self.b_cell,
                                   self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        if deterministic:
            self.using_dropout = False
        else:
            self.using_dropout = True
        cell_mask = self._srng.binomial((num_batch, self.num_units),
                                        p=T.constant(1) - self.p,
                                        dtype=floatX)

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            if self.use_layer_norm:
                gates = self.layer_norm(input=gates,
                                        alpha=self.alpha_gate,
                                        beta=self.beta_gate)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
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
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            if self.use_layer_norm:
                _cell = self.layer_norm(input=cell,
                                        alpha=self.alpha_cell,
                                        beta=self.beta_cell)
            else:
                _cell = cell

            hid = outgate*self.nonlinearity(_cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [cell_mask, W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.use_layer_norm:
            non_seqs +=[self.alpha_gate,
                        self.alpha_cell,
                        self.beta_gate,
                        self.beta_cell]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out

class BiDirLSTMLayer(MergeLayer):
    def __init__(self,
                 incoming,
                 num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 fwd_cell_init=init.Constant(0.),
                 fwd_hid_init=init.Constant(0.),
                 bwd_cell_init=init.Constant(0.),
                 bwd_hid_init=init.Constant(0.),
                 dropout_ratio=0.2,
                 use_layer_norm=True,
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
        self.fwd_hid_init_incoming_index = -1
        self.fwd_cell_init_incoming_index = -1
        self.bwd_hid_init_incoming_index = -1
        self.bwd_cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(fwd_hid_init, Layer):
            incomings.append(fwd_hid_init)
            self.fwd_hid_init_incoming_index = len(incomings)-1
        if isinstance(fwd_cell_init, Layer):
            incomings.append(fwd_cell_init)
            self.fwd_cell_init_incoming_index = len(incomings)-1
        if isinstance(bwd_hid_init, Layer):
            incomings.append(bwd_hid_init)
            self.bwd_hid_init_incoming_index = len(incomings)-1
        if isinstance(bwd_cell_init, Layer):
            incomings.append(bwd_cell_init)
            self.bwd_cell_init_incoming_index = len(incomings)-1

        super(BiDirLSTMLayer, self).__init__(incomings, **kwargs)

        self.fwd_lstm_layer = LSTMLayer(incoming=incoming,
                                        num_units=num_units,
                                        ingate=ingate,
                                        forgetgate=forgetgate,
                                        cell=cell,
                                        outgate=outgate,
                                        nonlinearity=nonlinearity,
                                        cell_init=fwd_cell_init,
                                        hid_init=fwd_hid_init,
                                        dropout_ratio=dropout_ratio,
                                        use_layer_norm=use_layer_norm,
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

        self.bwd_lstm_layer = LSTMLayer(incoming=incoming,
                                        num_units=num_units,
                                        ingate=ingate,
                                        forgetgate=forgetgate,
                                        cell=cell,
                                        outgate=outgate,
                                        nonlinearity=nonlinearity,
                                        cell_init=bwd_cell_init,
                                        hid_init=bwd_hid_init,
                                        dropout_ratio=dropout_ratio,
                                        use_layer_norm=use_layer_norm,
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

        self.num_units = num_units*2

    def get_output_shape_for(self, input_shapes):
        num_samples = input_shapes[0][0]
        num_step = input_shapes[0][1]
        num_units = self.num_units

        return (num_samples, num_step, num_units)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input_dict = {}

        input = inputs[0]
        input_dict[self.input_layers[0]] = input

        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
            input_dict[self.input_layers[self.mask_incoming_index]] = mask

        if self.fwd_hid_init_incoming_index > 0:
            fwd_hid_init = inputs[self.fwd_hid_init_incoming_index]
            input_dict[self.input_layers[self.fwd_hid_init_incoming_index]] = fwd_hid_init
        if self.fwd_cell_init_incoming_index > 0:
            fwd_cell_init = inputs[self.fwd_cell_init_incoming_index]
            input_dict[self.input_layers[self.fwd_cell_init_incoming_index]] = fwd_cell_init

        if self.bwd_hid_init_incoming_index > 0:
            bwd_hid_init = inputs[self.bwd_hid_init_incoming_index]
            input_dict[self.input_layers[self.bwd_hid_init_incoming_index]] = bwd_hid_init
        if self.bwd_cell_init_incoming_index > 0:
            bwd_cell_init = inputs[self.bwd_cell_init_incoming_index]
            input_dict[self.input_layers[self.bwd_cell_init_incoming_index]] = bwd_cell_init


        output = get_output(self.output_layer,
                            inputs=input_dict,
                            deterministic=deterministic)

        return output

