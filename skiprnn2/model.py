'''Tensorflow implementation of models, libraries, functions, modules'''
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

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

