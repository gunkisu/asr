'''Mixer containing functions or building blocks'''
import numpy as np
import theano
import theano.tensor as tensor

from collections import OrderedDict
from mixer import binary_sigmoid
from mixer import custom_init
from mixer import glorot_init
from mixer import init_tparams
from mixer import orthogonal_init
from mixer import _p


def _affine(inputs, tparams, prefix, bias=True):
  # Calculate the total size of arguments on dimension 1
  shapes = [arg.shape for arg in inputs]
  _ndim = shapes[0].ndim
  for shape in shapes:
    if shape.ndim != _ndim:
      raise ValueError("Dimensions of inputs should match")
  logits = tensor.dot(tensor.concatenate(inputs, axis=inputs[0].ndim-1),
                      tparams[_p(prefix, 'W')])
  if bias:
    logits += tparams[_p(prefix, 'b')]
  return logits


class LinearCell(object):
  """Implementation of Linear layer"""
  def __init__(self, num_inputs, num_units, prefix='linear', bias=True,
               activation=None, input_is_int=False, block_init=False):
    self._num_units = num_units
    self._prefix = prefix
    self._bias = bias
    self._activation = activation
    # num_inputs can be a list of dimension of each inputs
    self._num_inputs = num_inputs
    self._input_is_int = input_is_int
    self._block_init = block_init
    self._init_params()

  def _init_params(self):
    params = OrderedDict()
    nins = self._num_inputs
    nout = self._num_units
    if type(nins) is list:
      if self._block_init:
        W = custom_init(nins[0], nout)
        for i, nin in enumerate(nins[1:]):
          W = np.concatenate([W, custom_init(nin, nout)], axis=0)
      else:
        total_inputs_dim = np.array(nins).sum()
        W = custom_init(total_inputs_dim, nout)
    else:
      W = custom_init(nins, nout)
    params[_p(self._prefix, 'W')] = W
    if self._bias:
      params[_p(self._prefix, 'b')] = np.zeros((nout,)).astype('float32')
    self._params = init_tparams(params)

  def __call__(self, inputs):
    if self._input_is_int:
      if type(inputs) is list:
        raise AssertionError("If inputs is int type, it should not be "
                             "provided as a list and as multiple numbers.")
      n_step = inputs.shape[0]
      if inputs.ndim in [2, 3]:
        batch_size = inputs.shape[1]
      else:
        batch_size = 1
      logits = self._params[_p(self._prefix, 'W')][inputs.flatten()]
      if inputs.ndim == 2:
        logits = logits.reshape((n_step, batch_size, -1))
      if self._bias:
        logits += self._params[_p(self._prefix, 'b')]
    elif type(inputs) is list:
      logits = _affine(inputs, self._params, self._prefix, self._bias)
    else:
      logits = _affine([inputs], self._params, self._prefix, self._bias)
    if self._activation is None:
      return logits
    return self._activation(logits)


def _lstm_gates(logits, num_units):
    """Split logits into inputs, forget, output and candidate
    logits and apply appropriate activation functions.
    _inputs: inputs gates, _forget: forget gates,
    _output: output gates,  _cell: cell candidates
    """
    _inputs, _forget, _output, _cell = \
            theano.tensor.split(logits, [num_units] * 4, 4, 1)
    _inputs = tensor.nnet.sigmoid(_inputs)
    _forget = tensor.nnet.sigmoid(_forget)
    _output = tensor.nnet.sigmoid(_output)
    _cell = tensor.tanh(_cell)
    return _inputs, _forget, _output, _cell


class LSTMModule(object):
  """Implementation of LSTM module"""
  def __init__(self, num_inputs, num_units, prefix):
    self._prefix = prefix
    rnn_cell = LSTMCell(num_inputs, num_units, self._prefix)
    self._rnn_cell = rnn_cell
    self._params = rnn_cell._params

  def __call__(self, inputs, init_state):
    # State is [2, batch_size, n_hidden]
    rnn_list_state = [init_state[0], init_state[1]]
    # Define an LSTM cell with Theano
    outputs, updates = theano.scan(self._rnn_cell,
                                   sequences=[inputs],
                                   outputs_info=rnn_list_state,
                                   name=self._prefix)
    last_state = tensor.stack([outputs[0][-1], outputs[1][-1]], axis=0)
    return outputs, last_state


class HMLSTMModule(object):
  """Implementation of Hierarchical Multiscale LSTM module"""
  def __init__(self, num_inputs, num_units, prefix, use_impl_type='base'):
    self._prefix = prefix
    self._use_impl_type = use_impl_type
    # Define an LSTM cell with Theano
    if self._use_impl_type == 'base':
      rnn_cell = HMLSTMCell(num_inputs, num_units, self._prefix)
    else:
      raise ValueError("Not a valid impl type")
    self._rnn_cell = rnn_cell
    self._params = rnn_cell._params

  def __call__(self, inputs, init_state, init_boundary):
    # State is [n_layers, 2, batch_size, n_hidden]
    # Boundary is [n_layers-1, batch_size]
    rnn_list_state = self._unpack_state(init_state, init_boundary)
    outputs, updates = theano.scan(self._rnn_cell,
                                   sequences=[inputs],
                                   outputs_info=rnn_list_state,
                                   name=self._prefix)
    last_boundary = tensor.stack([outputs[-2][-1], outputs[-1][-1]], axis=0)
    last_state = tensor.stack(
            [tensor.stack([outputs[i][-1], outputs[i+1][-1]], axis=0)
             for i in xrange(0, 6, 2)],axis=0)
    return outputs, last_state, last_boundary

  def _unpack_state(self, init_state, init_boundary):
    list_state = []
    for i in xrange(3):
      list_state.append(init_state[i][0])
      list_state.append(init_state[i][1])
    return list_state + [init_boundary[0], init_boundary[1]]


class LSTMCell(object):
  """LSTM recurrent network cell"""
  def __init__(self, num_inputs, num_units, prefix='lstm'):
    """Initialize the LSTM cell.
    Args:
      num_inputs: int, the number of units of input,
      num_units: int, the number of units in the LSTM cell.
      prefix: str, name space of the object.
    """
    self._prefix = prefix
    self._num_inputs = num_inputs
    self._num_units = num_units
    self._init_params()

  def _init_params(self):
    params = OrderedDict()
    nin = self._num_inputs
    nout = self._num_units
    # First RNN
    W = self.custom_block_initializer(nin, nout)
    b = np.zeros((4 * nout,)).astype('float32')
    params[_p(self._prefix, 'W')] = W
    params[_p(self._prefix, 'b')] = b
    self._params = init_tparams(params)

  def __call__(self, inputs, h, c):
    """Long Short-Term Memory (LSTM) cell"""
    # Parameters of gates are concatenated into one multiply for efficiency.
    logits = _affine([inputs, h], self._params, self._prefix)
    i, f, o, j = _lstm_gates(logits, self._num_units)
    # Update the states
    new_c = c * f + i * j
    new_h = o * tensor.tanh(new_c)
    # Update the returns
    return new_h, new_c

  def custom_recurrent_initializer(self, num_units):
    x = orthogonal_init(num_units)
    U = np.concatenate([x] * 4, axis=1)
    return U

  def custom_block_initializer(self, num_input_units, num_units):
    """Custom weight initalizer for LSTM using numpy arrays.
    Block initialization for RNNs.
    Args:
      num_input_units: number of input units
      num_units: number of hidden units, assume equal for every layer
    """
    x = custom_init(num_input_units, num_units)
    W = np.concatenate([x] * 4, axis=1)
    U = self.custom_recurrent_initializer(num_units)
    W = np.vstack([W, U])
    return W


class HMLSTMCell(object):
  """Hierarchical Multiscale LSTM recurrent network cell.
  Three-layered, use binary straight-through.
  """
  def __init__(self, num_inputs, num_units, prefix='hm_lstm',
               is_topdown2forget_gate=False):
    """Initialize the Hierarchical Multiscale LSTM cell.
    Args:
      num_inputs: int, the number of units of input,
      num_units: int, the number of units in the HM-LSTM cell.
      prefix: str, name space of the object.
      to forget gate connection.
    """
    self._prefix = prefix
    self._num_inputs = num_inputs
    self._num_units = num_units
    self._init_params()

  def _init_params(self):
    params = OrderedDict()
    nin = self._num_inputs
    nout = self._num_units
    # First RNN
    W = self.custom_block_initializer(nin, nout, is_topdown=True)
    b = np.zeros((4 * nout + 1,)).astype('float32')
    params[_p(self._prefix, 'first_W')] = W
    params[_p(self._prefix, 'first_b')] = b
    # Second RNN
    W = self.custom_block_initializer(nout, nout, is_topdown=True)
    b = np.zeros((4 * nout + 1,)).astype('float32')
    params[_p(self._prefix, 'second_W')] = W
    params[_p(self._prefix, 'second_b')] = b
    # Third RNN
    W = self.custom_block_initializer(nout, nout, is_z=False)
    b = np.zeros((4 * nout,)).astype('float32')
    params[_p(self._prefix, 'third_W')] = W
    params[_p(self._prefix, 'third_b')] = b
    self._params = init_tparams(params)

  def __call__(self, inputs, h_1, c_1, h_2, c_2, h_3, c_3, z_1, z_2):
    """Hierarchical Multiscale Long Short-Term Memory (HM-LSTM) cell.
    State is a tuple of the first-layer c_1 and h_1, the second-layer
    c_2 and h_2, the third-layer c_3 and h_3, the first-layer boundary
    detector z_1 and the second-layer boundary detector z_2 at previous
    time step.
    """
    z_1 = z_1[:, None]
    z_2 = z_2[:, None]
    # First RNN
    prefix = _p(self._prefix, 'first')
    logits_1 = self.f_hmlstm(inputs, h_1, z_1 * h_2, prefix=prefix)
    # Split and apply activation
    f_1, i_1, o_1, j_1 = _lstm_gates(logits_1[:, :-1], self._num_units)
    # Update the states
    new_c_1 = (1. - z_1) * f_1 * c_1 + i_1 * j_1
    new_h_1 = o_1 * tensor.tanh(new_c_1)
    # Straight-through estimation
    new_z_1 = binary_sigmoid(logits_1[:, -1])[:, None]
    # Second RNN
    prefix = _p(self._prefix, 'second')
    logits_2 = self.f_hmlstm(new_z_1 * new_h_1, h_2, z_2 * h_3, prefix=prefix)
    # Split and apply activation
    f_2, i_2, o_2, j_2 = _lstm_gates(logits_2[:, :-1], self._num_units)
    # Update the states
    is_copy = (1. - z_2) * (1. - new_z_1)
    is_update = (1. - z_2) * new_z_1
    gated_j_2 = i_2 * j_2
    new_c_2 = gated_j_2 + is_copy * (c_2 - gated_j_2) + is_update * (f_2 * c_2)
    new_h_2 = is_copy * h_2 + (1. - is_copy) * o_2 * tensor.tanh(new_c_2)
    # Straight-through estimation
    new_z_2 = binary_sigmoid(logits_2[:, -1])[:, None]
    # Third RNN
    prefix = _p(self._prefix, 'third')
    logits_3 = self.f_hmlstm(new_z_2 * new_h_2, h_3, prefix=prefix,
                             is_last_layer=True)
    # Split and apply activation
    f_3, i_3, o_3, j_3 = _lstm_gates(logits_3, self._num_units)
    # Update the states
    new_c_3 = f_3 * c_3 + i_3 * j_3
    new_c_3 = new_z_2 * new_c_3 + (1. - new_z_2) * c_3
    new_h_3 = o_3 * tensor.tanh(new_c_3)
    # Update the returns
    new_z_1 = new_z_1.flatten()
    new_z_2 = new_z_2.flatten()
    return new_h_1, new_c_1, new_h_2, new_c_2, new_h_3, new_c_3, new_z_1, \
            new_z_2

  def f_hmlstm(self, h_below, h_before, h_above=None, prefix=None,
               is_last_layer=False):
    if is_last_layer:
      logits = _affine([h_below, h_before], self._params, prefix)
      return logits
    logits = _affine([h_below, h_before, h_above], self._params, prefix)
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
