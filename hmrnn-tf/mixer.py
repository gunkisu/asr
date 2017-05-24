import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


# Misc
def gen_mask(x, max_seq_len):
  n_step = x.shape[0]
  if n_step != max_seq_len:
    padded_x = np.zeros((max_seq_len, x.shape[1]))
    padded_x[:n_step] = x
    x = padded_x.astype('int64')
    x_mask = np.zeros((max_seq_len, x.shape[1])).astype(np.float32)
    x_mask[:n_step] = 1.
  else:
    x_mask = np.ones((x.shape[0], x.shape[1])).astype(np.float32)
  return x, x_mask


def insert_item2dict(ref_dict, key, value):
  try:
    ref_dict[key].append(value)
  except:
    ref_dict[key] = [value]


def nats2bits(x):
  return x / math.log(2)


def save_npz2(file_name, param_dict):
  np.savez(file_name, **param_dict)


# Initialization
def custom_init(nin, nout=None, scale=0.01, orthogonal=True):
  if nout is None:
    nout = nin
  if nout == nin and orthogonal:
    x = orthogonal_init(nin)
  else:
    x = glorot_init(nin, nout)
  return x


def glorot_init(nin, nout=None, uniform=True):
  if nout is None:
    nout = nin
  if uniform:
    scale = np.sqrt(6.0 / (nin + nout))
    x = uniform_init(nin, nout, scale)
  else:
    scale = np.sqrt(3.0 / (nin + nout))
    x = normal_init(nin, nout, scale)
  return x


def he_init(nin, nout=None, uniform=True):
  if nout is None:
    nout = nin
  scale = np.sqrt(1.0 / nin)
  if uniform:
    x = uniform_init(nin, nout, scale)
  else:
    x = normal_init(nin, nout, scale)
  return x


def normal_init(nin, nout=None, scale=0.01):
  x = scale * np.random.normal(loc=0.0, scale=1.0, size=(nin, nout))
  return x.astype(np.float32)


def orthogonal_init(nin):
  x = np.random.normal(0.0, 1.0, (nin, nin))
  u, _, v = np.linalg.svd(x, full_matrices=False)
  return u.astype(np.float32)


def uniform_init(nin, nout=None, scale=0.01):
  x = np.random.uniform(size=(nin, nout), low=-scale, high=scale)
  return x.astype(np.float32)


# Tensorflow
def add_summary_value(summary, tag, simple_value):
  value = summary.value.add()
  value.tag = tag
  value.simple_value = simple_value


# Tensorflow Activation
def hard_sigmoid(x, scale=1.):
  return tf.clip_by_value((scale * x + 1.) / 2., clip_value_min=0,
                          clip_value_max=1)


# Tensorflow Op
def binary_round(x):
  """Rounds a tensor whose values are in [0, 1] to a tensor
  with values in {0, 1}, using the straight through estimator
  to approximate the gradient.
  """
  g = tf.get_default_graph()
  with ops.name_scope("binary_round") as name:
    with g.gradient_override_map({"Round": "Identity"}):
      return tf.round(x, name=name)


def binary_sigmoid(x, slope_tensor=None):
  """Straight through hard sigmoid.
  Hard sigmoid followed by the step function.
  """
  if slope_tensor is None:
    slope_tensor = tf.constant(1.0)
  p = hard_sigmoid(x, slope_tensor)
  return binary_round(p)


def scatter_add_tensor(ref, indices, updates, name=None):
  """
  Adds sparse updates to a variable reference.

  This operation outputs ref after the update is done. This makes it easier
  to chain operations that need to use the reset value.

  Duplicate indices are handled correctly: if multiple indices reference
  the same location, their contributions add.

  Requires updates.shape = indices.shape + ref.shape[1:].
  Args:
    param ref: A Tensor. Must be one of the following types: float32, float64,
      int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8,
      quint8, qint32, half.
    param indices: A Tensor. Must be one of the following types: int32, int64.
      A tensor of indices into the first dimension of ref.
    param updates: A Tensor. Must have the same dtype as ref.
      A tensor of updated values to add to ref
    param name: A name for the operation (optional).
  Return:
    Same as ref. Returned as a convenience for operations that want
    to use the updated values after the update is done.
  """
  with tf.name_scope(name, 'scatter_add_tensor',
                     [ref, indices, updates]) as scope:
    ref = tf.convert_to_tensor(ref, name='ref')
    indices = tf.convert_to_tensor(indices, name='indices')
    updates = tf.convert_to_tensor(updates, name='updates')
    ref_shape = tf.shape(ref, out_type=indices.dtype, name='ref_shape')
    scattered_updates = tf.scatter_nd(indices, updates, ref_shape,
                                      name='scattered_updates')
    with tf.control_dependencies(
            [tf.assert_equal(ref_shape,
                             tf.shape(scattered_updates,
                                      out_type=indices.dtype))]):
      output = tf.add(ref, scattered_updates, name=scope)
  return output


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y
