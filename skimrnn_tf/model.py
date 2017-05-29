import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid


def _affine(args, output_size, bias=True, init_W=None):
    # Calculate the total size of arguments on dimension 1
    total_arg_size = 0
    shapes = [arg.get_shape() for arg in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value
    # Set data type
    dtype = args[0].dtype
    # Compute linear part
    _scope = tf.get_variable_scope()
    with tf.variable_scope(_scope) as outer_scope:
        if init_W is not None:
            W = tf.get_variable('W', initializer=init_W, dtype=dtype)
        else:
            W = tf.get_variable('W', [total_arg_size, output_size], dtype=dtype)
        tf.add_to_collection('weights', W)
        tf.add_to_collection('vars', W)
        if len(args) == 1:
            logits = math_ops.matmul(args[0], W)
        else:
            logits = math_ops.matmul(array_ops.concat(args, 1), W)
        if not bias:
            return logits
        b = tf.get_variable('b',
                            [output_size],
                            dtype=dtype,
                            initializer=init_ops.constant_initializer(0.0, dtype=dtype))
        tf.add_to_collection('vars', b)
        return nn_ops.bias_add(logits, b)


def _lstm_gates(logits, num_splits=4, axis=1, activation=tanh, forget_bias=0.0):
    _input, _forget, _output, _cell = array_ops.split(value=logits,
                                                      num_or_size_splits=num_splits,
                                                      axis=axis)
    _input = sigmoid(_input)
    _forget = sigmoid(_forget + forget_bias)
    _output = sigmoid(_output)
    _cell = activation(_cell)
    return _input, _forget, _output, _cell


_SkimLSTMStateTuple = namedtuple("SkimLSTMStateTuple", ("c",  # cell state (vector, matrix)
                                                        "h",  # hidden state (vector, matrix)
                                                        "s",  # skim counter (integer, vector)
                                                        "r")) # read counter (integer, vector)


class SkimLSTMStateTuple(_SkimLSTMStateTuple):
    __slots__ = ()
    @property
    def dtype(self):
        (c, h, s) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(c.dtype), str(h.dtype)))
        return c.dtype


class SkimLSTMCell(RNNCell):
    def __init__(self,
                 num_units, # hidden units
                 max_skims=5, # possible max skim steps (number of actions)
                 min_reads=2, # minimum read steps after skimming
                 forget_bias=0.0, # forget gate bias
                 activation=tanh,
                 reuse=None):
        self._num_units = num_units
        self._max_skims = max_skims
        self._min_reads = float(min_reads)
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        cell_size = self._num_units
        hid_size = self._num_units
        cntr_size = 1
        return SkimLSTMStateTuple(cell_size, hid_size, cntr_size, cntr_size)

    @property
    def output_size(self):
        return self._num_units, self._max_skims, 1, 1, 1

    def __call__(self,
                 inputs,
                 state,
                 scope=None):
        with _checked_scope(self, scope or "skim_lstm_cell", reuse=self._reuse):
            # get input data and mask
            input_data = inputs[:, :-1]
            input_mask = tf.expand_dims(inputs[:, -1], axis=-1)

            # get previous states
            prev_c, prev_h, skim_cntr, read_cntr = state

            # update mask based on mask and read_counter
            # if read counter is larger than 0, need to update
            read_mask = tf.to_float((read_cntr * input_mask) > 0, name='read_mask')

            # action mask based on mask and read_counter
            # if read counter is 1 that now requires action
            action_mask = tf.to_float((tf.to_float(tf.equal(read_cntr,1.0)) * input_mask) > 0, name='action_mask')

            # init read mask based on mask and skim_counter
            # if skim counter is 1 that next should be read
            init_mask = tf.to_float((tf.to_float(tf.equal(skim_cntr, 1.0)) * input_mask) > 0, name='init_mask')
            skim_mask = tf.to_float((skim_cntr * input_mask) > 0, name='skim_mask')

            # reduce read counter
            new_read_cntr = tf.maximum(x=read_cntr - read_mask, y=0.0, name='new_read_cntr')
            new_skim_cntr = tf.maximum(x=skim_cntr - skim_mask, y=0.0, name='new_skim_cntr')

            # first update states
            # compute gate
            with vs.variable_scope("gate"):
                gate_logits = _affine([input_data, prev_h], 4 * self._num_units)
            i, f, o, j = _lstm_gates(gate_logits, forget_bias=self._forget_bias)
            new_c = prev_c * f + i * j
            new_h = o * self._activation(new_c)
            new_c = new_c * read_mask + prev_c * (1. - read_mask)
            new_h = new_h * read_mask + prev_h * (1. - read_mask)

            # compute skim action
            with vs.variable_scope("action"):
                action_input = tf.stop_gradient(new_h)
                action_logit = _affine([action_input, ], self._max_skims)
            action_logprob = action_logit - tf.log(tf.reduce_sum(input_tensor=tf.exp(action_logit),
                                                                 axis=1,
                                                                 keep_dims=True))
            action_sample = tf.to_float(tf.multinomial(logits=action_logit, num_samples=1))

            # update skim counter
            new_skim_cntr += (action_sample + 1.0) * action_mask

            # update skim counter
            new_read_cntr += self._min_reads * init_mask

            # return
            # 1) outputs (it will be return over sequence as output)
            outputs = new_h, action_logprob, action_sample, read_mask, action_mask

            # 2) states (it will be return for just next step)
            new_state = SkimLSTMStateTuple(new_c, new_h, new_skim_cntr, new_read_cntr)

            return outputs, new_state


class SkimLSTMModule(object):
    def __init__(self,
                 num_units,
                 max_skims=5,
                 min_reads=1,
                 forget_bias=1.0,
                 activation=tanh):
        self._num_units = num_units
        self._max_skims = max_skims
        self._min_reads = min_reads
        self._forget_bias = forget_bias
        self._activation = activation

    def __call__(self,
                 inputs,
                 init_state,
                 use_bidir=False,
                 scope=None):
        # Forward
        # init state
        fwd_init_state = SkimLSTMStateTuple(init_state[0],
                                            init_state[0],
                                            tf.zeros_like(init_state[1]),
                                            tf.ones_like(init_state[1])*self._min_reads)

        # init cell
        fwd_rnn_cell = SkimLSTMCell(num_units=self._num_units,
                                    max_skims=self._max_skims,
                                    min_reads=self._min_reads,
                                    forget_bias=self._forget_bias,
                                    activation=self._activation)
        # init loop
        fwd_outputs, _ = tf.nn.dynamic_rnn(cell=fwd_rnn_cell,
                                           inputs=inputs,
                                           initial_state=fwd_init_state,
                                           time_major=True,
                                           scope='fwd_dir')

        fwd_hid_data, fwd_act_lgp_seq, fwd_act_sample_seq, fwd_read_mask, fwd_act_mask = fwd_outputs

        fwd_act_lgp = fwd_act_lgp_seq * tf.one_hot(indices=tf.to_int32(tf.squeeze(fwd_act_sample_seq)),
                                                   depth=self._max_skims) * fwd_act_mask

        # Backward
        if use_bidir:
            # init state
            bwd_init_state = SkimLSTMStateTuple(init_state[0],
                                                init_state[1],
                                                tf.zeros_like(init_state[2]),
                                                tf.ones_like(init_state[2])*self._min_reads)

            # init cell
            bwd_rnn_cell = SkimLSTMCell(num_units=self._num_units,
                                        max_skims=self._max_skims,
                                        min_reads=self._min_reads,
                                        forget_bias=self._forget_bias,
                                        activation=self._activation)
            # init loop
            bwd_outputs, _ = tf.nn.dynamic_rnn(cell=bwd_rnn_cell,
                                               inputs=tf.reverse(inputs, axis=[0]),
                                               initial_state=bwd_init_state,
                                               time_major=True,
                                               scope='bwd_dir')

            bwd_hid_data, bwd_act_lgp_seq, bwd_act_sample_seq, bwd_read_mask, bwd_act_mask = bwd_outputs

            hid_data = tf.concat([fwd_hid_data, tf.reverse(bwd_hid_data, axis=[0])], axis=-1)
            seq_mask = fwd_read_mask * tf.reverse(bwd_read_mask, axis=[0])

            bwd_act_lgp = tf.reverse(bwd_act_lgp_seq, axis=[0])*tf.one_hot(indices=tf.to_int32(tf.squeeze(tf.reverse(bwd_act_sample_seq, axis=[0]))), depth=self._max_skims)*tf.reverse(bwd_act_mask, axis=[0])

            outputs = hid_data, seq_mask, fwd_act_lgp, bwd_act_lgp
        else:
            hid_data = fwd_hid_data
            seq_mask = fwd_read_mask
            act_lgp = fwd_act_lgp

            outputs = hid_data, seq_mask, act_lgp
        return outputs

class LinearCell(object):
    """Implementation of Linear layer"""
    def __init__(self,
                 num_units,
                 bias=True,
                 activation=None,
                 num_inputs=None,
                 input_is_int=False):
        self._num_units = num_units
        self._activation = activation
        self._bias = bias
        self._num_inputs = num_inputs
        self._input_is_int = input_is_int

    def __call__(self,
                 inputs,
                 scope='linear',
                 init_W=None):
        if self._input_is_int:
            if self._num_inputs is None:
                raise ValueError("Number of units of previous layer should "
                                 "be given to determine the weights size.")
            with tf.variable_scope(scope) as scope:
                if init_W is not None:
                    W = tf.get_variable('W', initializer=init_W, dtype=tf.float32)
                else:
                    W = tf.get_variable('W', [self._num_inputs, self._num_units],
                                        dtype=tf.float32)
                tf.add_to_collection('weights', W)
                tf.add_to_collection('vars', W)

                logits = tf.reshape(tf.gather(W, tf.reshape(inputs, [-1])),
                                    [-1, self._num_units])

                if self._bias:
                    b = tf.get_variable('b', [self._num_units],
                                        initializer=init_ops.constant_initializer(0.0))
                    tf.add_to_collection('vars', b)
                    logits = nn_ops.bias_add(logits, b)
        elif type(inputs) is list:
            logits = _affine(inputs, self._num_units, self._bias, init_W=init_W)
        else:
            logits = _affine([inputs], self._num_units, self._bias, init_W=init_W)
        if self._activation is None:
            return logits
        return self._activation(logits)