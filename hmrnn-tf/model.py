'''Tensorflow implementation of models, libraries, functions, modules'''
import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from mixer import binary_round
from mixer import binary_sigmoid
from mixer import custom_init
from mixer import glorot_init
from mixer import orthogonal_init

class LSTMModule(object):
  """Implementation of LSTM module"""
  def __init__(self, num_units):
    self._num_units = num_units

  def __call__(self, inputs, init_state, one_step=False):
    rnn_tuple_state = tf.contrib.rnn.LSTMStateTuple(init_state[0],
                                                    init_state[1])
    # Define an LSTM cell with Tensorflow
    rnn_cell = LSTMCell(self._num_units)

    if one_step:
      with tf.variable_scope('rnn'):
        outputs, states = rnn_cell(inputs=inputs,
                                   state=rnn_tuple_state)
      last_state = tf.stack([states[0], states[1]], axis=0, name='one_step_stack')
    else:
      outputs, states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                          inputs=inputs,
                                          initial_state=rnn_tuple_state)
#                                          time_major=True)
      last_state = tf.stack([states[0], states[1]], axis=0, name='dynamic_stack')
    return outputs, last_state

class StackLSTMModule(object):
  """Implementation of Stacked LSTM module"""
  def __init__(self, num_units, num_layers=1):
    self._num_units = num_units
    self._num_layers = num_layers

  def __call__(self,
               inputs,
               init_state,
               one_step=False):
    if isinstance(init_state, tuple) or isinstance(init_state, list):
      rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(init_state[l][0], init_state[l][1])
                               for l in range(self._num_layers)])
    else:
      rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(init_state[0], init_state[1])
                               for l in range(self._num_layers)])

    # Define an LSTM cell with Tensorflow
    rnn_cell_list = []
    for l in range(self._num_layers):
      with tf.variable_scope('lstm_{}'.format(l)) as vs:
        rnn_cell_list.append(LSTMCell(self._num_units))

    # Stack
    stack_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell_list, state_is_tuple=True)

    if one_step:
      with tf.variable_scope('rnn'):
        outputs, states = stack_rnn_cell(inputs=inputs, state=rnn_tuple_state)
      stack_name = 'one_step_stack'
    else:
      outputs, states = tf.nn.dynamic_rnn(cell=stack_rnn_cell,
                                          inputs=inputs,
                                          initial_state=rnn_tuple_state)
      stack_name = 'dynamic_stack'
    last_state_list = [tf.stack([states[l][0], states[l][1]], axis=0, name=stack_name) for l in range(self._num_layers)]
    return outputs, last_state_list

def _affine(args, output_size, bias=True, scope=None, init_W=None):
  # Calculate the total size of arguments on dimension 1
  total_arg_size = 0
  shapes = [arg.get_shape() for arg in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value
  # Set data type
  dtype = args[0].dtype
  # Compute linear part
  _scope = tf.get_variable_scope()
  with tf.variable_scope(_scope) as outer_scope:
    with tf.variable_scope(scope) as inner_scope:
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
      b = tf.get_variable('b', [output_size], dtype=dtype,
        initializer=init_ops.constant_initializer(0.0, dtype=dtype))
      tf.add_to_collection('vars', b)
  return nn_ops.bias_add(logits, b)


class LinearCell(object):
  """Implementation of Linear layer"""
  def __init__(self, num_units, bias=True, activation=None, num_inputs=None,
               input_is_int=False):
    self._num_units = num_units
    self._activation = activation
    self._bias = bias
    self._num_inputs = num_inputs
    self._input_is_int = input_is_int

  def __call__(self, inputs, scope='linear', init_W=None):
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
        if self._bias:
          b = tf.get_variable('b', [self._num_units],
                              initializer=init_ops.constant_initializer(0.0))
          tf.add_to_collection('vars', b)
      logits = tf.reshape(tf.gather(W, tf.reshape(inputs, [-1])),
                          [-1, self._num_units])
      if self._bias:
        logits = nn_ops.bias_add(logits, b)
    elif type(inputs) is list:
      logits = _affine(inputs, self._num_units, self._bias, scope=scope,
                       init_W=init_W)
    else:
      logits = _affine([inputs], self._num_units, self._bias, scope=scope,
                       init_W=init_W)
    if self._activation is None:
      return logits
    return self._activation(logits)


def _lstm_gates(logits, num_splits=4, axis=1, activation=tanh, forget_bias=0.0):
    """Split logits into input, forget, output and candidate
    logits and apply appropriate activation functions.
    _input: input gates, _forget: forget gates,
    _output: output gates,  _cell: cell candidates
    """
    _input, _forget, _output, _cell = \
            array_ops.split(value=logits, num_or_size_splits=num_splits,
                            axis=axis)
    _input = sigmoid(_input)
    _forget = sigmoid(_forget + forget_bias)
    _output = sigmoid(_output)
    _cell = activation(_cell)
    return _input, _forget, _output, _cell

class HMLSTMModule(object):
  """Implementation of Hierarchical Multiscale LSTM module"""
  def __init__(self, num_units, use_impl_type='base'):
    self._num_units = num_units
    self._use_impl_type = use_impl_type

  def __call__(self, inputs, init_state, init_boundary, scope=None):
    state_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.contrib.rnn.LSTMStateTuple(
         state_list[idx][0], state_list[idx][1])
         for idx in range(len(state_list))] +
        [init_boundary[0], init_boundary[1]])
    # Define an LSTM cell with Tensorflow
    if self._use_impl_type == 'base':
      rnn_cell = HMLSTMCell(self._num_units)
    else:
      raise ValueError("Not a valid impl type")
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs,
                                        initial_state=rnn_tuple_state,
                                        time_major=True)
    # Stack last boundaries of vectors to a matrix
    last_boundary = tf.stack([states[-2], states[-1]], axis=0)
    # Stack last states of matrices to a 4-D tensor
    last_state = tf.stack([tf.stack([_state[0], _state[1]], axis=0)
                           for _state in states[:3]], axis=0)
    return outputs, last_state, last_boundary


class LSTMCell(RNNCell):
  """LSTM recurrent network cell.
  """
  def __init__(self, num_units, forget_bias=0.0, activation=tanh):
    """Initialize the LSTM cell.
    Args:
      num_units: int, the number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    return LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope='lstm'):
    """Long Short-Term Memory (LSTM) cell.
    """
    # Parameters of gates are concatenated into one multiply for efficiency.
    c, h = state
    logits = _affine([inputs, h], 4 * self._num_units, scope=scope)
    i, f, o, j = _lstm_gates(logits, forget_bias=self._forget_bias)
    # Update the states
    new_c = c * f + i * j
    new_h = o * self._activation(new_c)
    # Update the returns
    new_state = LSTMStateTuple(new_c, new_h)
    return new_h, new_state


class HMLSTMCell(RNNCell):
  """Hierarchical Multiscale LSTM recurrent network cell.
  Three-layered, use binary straight-through.
  """
  def __init__(self, num_units, forget_bias=0.0, activation=tanh):
    """Initialize the Hierarchical Multiscale LSTM cell.
    Args:
      num_units: int, the number of units in the HM-LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      to forget gate connection.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units),
            LSTMStateTuple(self._num_units, self._num_units),
            LSTMStateTuple(self._num_units, self._num_units))

  @property
  def output_size(self):
    return (self._num_units, self._num_units, self._num_units, 1, 1)

  def __call__(self, inputs, state, scope='hm_lstm'):
    """Hierarchical Multiscale Long Short-Term Memory (HM-LSTM) cell.
    State is a tuple of the first-layer c_1 and h_1, the second-layer
    c_2 and h_2, the third-layer c_3 and h_3, the first-layer boundary
    detector z_1 and the second-layer boundary detector z_2 at previous
    time step.
    """
    # States from previous time step
    (c_1, h_1), (c_2, h_2), (c_3, h_3), z_1, z_2 = state
    # First RNN
    logits_1 = self.f_hmlstm(inputs, h_1, z_1 * h_2, scope='first_rnn')
    # Split and apply activation
    i_1, f_1, o_1, j_1 = _lstm_gates(logits_1[:, :-1])
    # Update the states
    is_update = 1. - z_1
    # if is_flush: z_1 == 1 only i_1 * j_1 is passed
    new_c_1 = is_update * f_1 * c_1 + i_1 * j_1
    new_h_1 = o_1 * tanh(new_c_1)
    # Straight-through estimation
    new_z_1 = binary_sigmoid(logits_1[:, -1])
    new_z_1 = tf.expand_dims(new_z_1, axis=1)
    # Second RNN
    logits_2 = self.f_hmlstm(new_z_1 * new_h_1, h_2, z_2 * h_3,
                             scope='second_rnn')
    # Split and apply activation
    i_2, f_2, o_2, j_2 = _lstm_gates(logits_2[:, :-1])
    # Update the states
    is_copy = (1. - z_2) * (1. - new_z_1)
    is_update = (1. - z_2) * new_z_1
    gated_j_2 = i_2 * j_2
    new_c_2 = gated_j_2 + is_copy * (c_2 - gated_j_2) + is_update * (f_2 * c_2)
    new_h_2 = is_copy * h_2 + (1. - is_copy) * o_2 * tanh(new_c_2)
    # Straight-through estimation
    new_z_2 = binary_sigmoid(logits_2[:, -1])
    new_z_2 = tf.expand_dims(new_z_2, axis=1)
    # Third RNN
    logits_3 = self.f_hmlstm(new_z_2 * new_h_2, h_3, scope='third_rnn',
                             is_last_layer=True)
    # Split and apply activation
    i_3, f_3, o_3, j_3 = _lstm_gates(logits_3)
    # Update the states
    new_c_3 = f_3 * c_3 + i_3 * j_3
    new_c_3 = new_z_2 * new_c_3 + (1. - new_z_2) * c_3
    new_h_3 = o_3 * tanh(new_c_3)
    # Update the returns
    new_state_1 = LSTMStateTuple(new_c_1, new_h_1)
    new_state_2 = LSTMStateTuple(new_c_2, new_h_2)
    new_state_3 = LSTMStateTuple(new_c_3, new_h_3)
    new_state = (new_state_1, new_state_2, new_state_3, new_z_1, new_z_2)
    return (new_h_1, new_h_2, new_h_3, new_z_1, new_z_2), new_state

  def f_hmlstm(self, h_below, h_before, h_above=None, scope=None,
               is_last_layer=False):
    if is_last_layer:
      W = self.custom_block_initializer(h_below.shape[1], self._num_units,
                                        is_z=False)
      logits = _affine([h_below, h_before], 4 * self._num_units,
                       scope=scope, init_W=W)
      return logits
    W = self.custom_block_initializer(h_below.shape[1], self._num_units,
                                      is_topdown=True)
    logits = _affine([h_below, h_before, h_above], 4 * self._num_units + 1,
                     scope=scope, init_W=W)
    return logits

  def custom_recurrent_initializer(self, num_units, is_z=True):
    x = orthogonal_init(num_units)
    U = np.concatenate([x] * 4, axis=1)
    if is_z:
      z = glorot_init(num_units, 1)
      U = np.concatenate([U, z], axis=1)
    return U

  def custom_block_initializer(self, num_input_units, num_units, is_z=True,
                               is_topdown=False):
    """Custom weight initalizer for HM-LSTM using numpy arrays.
    Block initialization for RNNs.
    Args:
      num_input_units: number of input units
      num_units: number of hidden units, assume equal for every layer
    """
    try:
      num_input_units = num_input_units.value
    except:
      pass
    x = custom_init(num_input_units, num_units)
    W = np.concatenate([x] * 4, axis=1)
    if is_z:
      z = glorot_init(num_input_units, 1)
      W = np.concatenate([W, z], axis=1)
    U = self.custom_recurrent_initializer(num_units, is_z)
    W = np.vstack([W, U])
    if is_topdown:
      W = np.vstack([W, U])
    return W
