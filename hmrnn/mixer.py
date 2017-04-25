'''Mixer containing functions or building blocks.
'''
import math
import numpy as np
import shutil
import theano
import theano.tensor as tensor
import warnings

from collections import OrderedDict
from theano.scalar.basic import UnaryScalarOp
from theano.scalar.basic import same_out_nocomplex
from theano.tensor.elemwise import Elemwise


# Misc
def feed_dict(summary, new_summary_value):
  for k, v in new_summary_value.items():
    try:
      summary[k].append(v)
    except:
      summary[k] = [v]


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


def itemlist(tparams):
  """Get the list of parameters: Note that tparams must be OrderedDict"""
  return [v for k, v in tparams.iteritems()]


def load_params(path, params):
  """Load parameters"""
  loaded_params = np.load(path)
  for k, v in params.iteritems():
    if k not in loaded_params.keys():
      warnings.warn('%s not in the archive' % k)
      continue
    params[k] = loaded_params[k]
  return params


def merge_dict(ref_dict, src_dict):
  for k, v in src_dict.items():
    ref_dict[k] = v
  return ref_dict


def nats2bits(x):
  return x / math.log(2)


def reset_state(state):
  for x in state:
    x.set_value(np.zeros(x.get_value().shape).astype('float32'))


def save_npz(file_name, global_step, epoch_step, batch_step, param_dict,
             summary_dict):
  #np.savez(file_name + '.tmp', global_step=global_step, epoch_step=epoch_step,
  #         batch_step=batch_step, **merge_dict(param_dict, summary_dict))
  #shutil.move(file_name + '.tmp', file_name)
  np.savez(file_name, global_step=global_step, epoch_step=epoch_step,
           batch_step=batch_step, **merge_dict(param_dict, summary_dict))


def save_npz2(file_name, param_dict):
  #np.savez(file_name + '.tmp', param_dict)
  #shutil.move(file_name + '.tmp', file_name)
  np.savez(file_name, **param_dict)


def _p(pp, name):
  """Make prefix-appended name"""
  return '%s_%s' % (pp, name)


def _slice(x, n, dim):
  """Utility function to slice a tensor"""
  if x.ndim == 1:
    return x[n*dim:(n+1)*dim]
  if x.ndim == 3:
    return x[:, :, n*dim:(n+1)*dim]
  return x[:, n*dim:(n+1)*dim]


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


# Theano
def castX(value):
  return theano._asarray(value, dtype=theano.config.floatX)


def init_tparams(params):
  """Initialize Theano shared variables according to the initial parameters"""
  tparams = OrderedDict()
  for k, p in params.iteritems():
    tparams[k] = theano.shared(params[k], name=k)
  return tparams

def init_tparams_with_restored_value(tparams, path):
  params = np.load(path)
  for k, v in tparams.iteritems():
    if k not in params.keys():
      warnings.warn('%s not in the archive' % k)
      continue
    tparams[k].set_value(params[k])
  return tparams


def sharedX(value, name=None, borrow=False, broadcastable=None):
  return theano.shared(castX(value), name=name, borrow=borrow,
                       broadcastable=broadcastable)


def unzip(zipped):
  """pull parameters from Theano shared variables"""
  new_params = OrderedDict()
  for k, v in zipped.iteritems():
    new_params[k] = v.get_value()
  return new_params


def zipp(params, tparams):
  """Push parameters to Theano shared variables"""
  for k, v in params.iteritems():
    tparams[k].set_value(v)


# Theano Optimizers
def gradient_clipping(grads, tparams, clip_c=1.0):
  g2 = 0.
  for g in grads:
    g2 += (g**2).sum()
  g2 = tensor.sqrt(g2)
  new_grads = []
  for p, g in zip(tparams.values(), grads):
    new_grads.append(tensor.switch(g2 > clip_c,
                                   g * (clip_c / g2),
                                   g))
  return new_grads


def adam(lr, tparams, grads, b1=0.9, b2=0.99, eps=1e-8):
  updates = OrderedDict()
  optparams = OrderedDict()
  optparams['i'] = np.float32(0.)
  for k, p in tparams.items():
    optparams[_p(k, 'm')] = p.get_value() * 0.
    optparams[_p(k, 'v')] = p.get_value() * 0.
  opt_tparams = init_tparams(optparams)
  i_t = opt_tparams['i'] + 1.
  fix1 = b1**i_t
  fix2 = b2**i_t
  lr_t = lr * tensor.sqrt(1. - fix2) / (1. - fix1)
  for (k, p), g in zip(tparams.items(), grads):
    m_t = b1 * opt_tparams[_p(k, 'm')] + (1. - b1) * g
    v_t = b2 * opt_tparams[_p(k, 'v')] + (1. - b2) * g**2
    g_t = lr_t * m_t / (tensor.sqrt(v_t) + eps)
    p_t = p - g_t
    updates[opt_tparams[_p(k, 'm')]] = m_t
    updates[opt_tparams[_p(k, 'v')]] = v_t
    updates[p] = p_t
  updates[opt_tparams['i']] = i_t
  return updates, opt_tparams


# Theano Op
class Round(UnaryScalarOp):

  def c_code(self, node, name, (x,), (z,), sub):
    return "%(z)s = round(%(x)s);" % locals()

  def grad(self, inputs, gout):
    (gz,) = gout
    return gz,


round_scalar = Round(same_out_nocomplex, name='round')
round = Elemwise(round_scalar)


def hard_sigmoid(x, scale=1.):
  return tensor.clip((scale * x + 1.) / 2., 0, 1)


def binary_sigmoid(x, scale=1.):
  return round(hard_sigmoid(x, scale))
