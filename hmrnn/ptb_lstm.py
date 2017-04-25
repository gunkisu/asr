'''Build a simple neural language model using LSTM-RNN.
'''
import argparse
import numpy as np
import os
import sys
import theano
import theano.tensor as tensor

from collections import OrderedDict
from itertools import izip
from mixer import adam
from mixer import gen_mask
from mixer import gradient_clipping
from mixer import itemlist
from mixer import merge_dict
from mixer import nats2bits
from mixer import reset_state
from mixer import sharedX
from model import LinearCell
from model import LSTMModule
from ptb_train import TrainModel
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def monitor(f_log_prob, FLAGS, valid_set, train_set=None, states=None):
  print("Start monitoring phase")
  returns = OrderedDict()
  if train_set is None:
    tr_nats = 0.0
    tr_bits = 0.0
  else:
    if states is not None:
      reset_state(states)
    _cost = 0
    _len = 0
    for x in train_set:
      x, x_mask = gen_mask(x, max_seq_len=FLAGS.max_seq_len)
      _tr_cost, _tr_cost_len = f_log_prob(x, x_mask)
      _cost += _tr_cost.sum()
      _len += _tr_cost_len.sum()
    tr_nats = _cost / _len
    tr_bits = nats2bits(tr_nats)
  returns['tr_nats'] = tr_nats
  returns['tr_bits'] = tr_bits
  if states is not None:
    reset_state(states)
  _cost = 0
  _len = 0
  for x in valid_set:
    x, x_mask = gen_mask(x, max_seq_len=FLAGS.max_seq_len)
    _val_cost, _val_cost_len = f_log_prob(x, x_mask)
    _cost += _val_cost.sum()
    _len += _val_cost_len.sum()
  val_nats = _cost / _len
  val_bits = nats2bits(val_nats)
  returns['val_nats'] = val_nats
  returns['val_bits'] = val_bits
  return returns


def build_graph(FLAGS):
  """Define training graph.
  """
  tparams = OrderedDict()
  trng = RandomStreams(
          np.random.RandomState(
              np.random.randint(1024)).randint(np.iinfo(np.int32).max))
  print("Building the computational graph")
  # Define bunch of shared variables
  init_state = np.zeros((3, 2, FLAGS.batch_size, FLAGS.n_hidden),
                        dtype=np.float32)
  tstate = sharedX(init_state, name='rnn_state')
  # Graph input
  inp = tensor.matrix('inp', dtype='int64')
  inp_mask = tensor.matrix('inp_mask', dtype='float32')
  inp.tag.test_value = np.zeros((FLAGS.max_seq_len, FLAGS.batch_size),
                                dtype='int64')
  inp_mask.tag.test_value = np.ones((FLAGS.max_seq_len, FLAGS.batch_size),
                                    dtype='float32')
  x, y = inp[:-1], inp[1:]
  y_mask = inp_mask[1:]
  # Define input embedding layer
  _i_embed = LinearCell(FLAGS.n_class, FLAGS.n_input_embed, prefix='i_embed',
                        bias=False, input_is_int=True)
  tparams = merge_dict(tparams, _i_embed._params)
  # Call input embedding layer
  h_i_emb_3d = _i_embed(x)
  # Define the first LSTM module
  _rnn_1 = LSTMModule(FLAGS.n_input_embed, FLAGS.n_hidden, prefix='lstm_1')
  tparams = merge_dict(tparams, _rnn_1._params)
  # Call the first LSTM module
  (h_rnn_1_3d, c_rnn_1_3d), last_state_1 = _rnn_1(h_i_emb_3d, tstate[0])
  # Define the second LSTM module
  _rnn_2 = LSTMModule(FLAGS.n_hidden, FLAGS.n_hidden, prefix='lstm_2')
  tparams = merge_dict(tparams, _rnn_1._params)
  # Call the second LSTM module
  (h_rnn_2_3d, c_rnn_2_3d), last_state_2 = _rnn_2(h_rnn_1_3d, tstate[1])
  # Define the third LSTM module
  _rnn_3 = LSTMModule(FLAGS.n_hidden, FLAGS.n_hidden, prefix='lstm_3')
  tparams = merge_dict(tparams, _rnn_3._params)
  # Call the third LSTM module
  (h_rnn_3_3d, c_rnn_3_3d), last_state_3 = _rnn_3(h_rnn_2_3d, tstate[2])
  # Define output gating layer
  _o_gate = LinearCell([FLAGS.n_hidden] * 3, 3, prefix='o_gate',
                       activation=tensor.nnet.sigmoid)
  tparams = merge_dict(tparams, _o_gate._params)
  # Call output gating layer
  h_o_gate = _o_gate([h_rnn_1_3d, h_rnn_2_3d, h_rnn_3_3d])
  # Define output embedding layer
  _o_embed = LinearCell([FLAGS.n_hidden] * 3, FLAGS.n_output_embed,
                        prefix='o_embed',
                        activation=tensor.nnet.relu)
  tparams = merge_dict(tparams, _o_embed._params)
  # Call output embedding layer
  h_o_embed = _o_embed([h_rnn_1_3d * h_o_gate[:, :, 0][:, :, None],
                        h_rnn_2_3d * h_o_gate[:, :, 1][:, :, None],
                        h_rnn_3_3d * h_o_gate[:, :, 2][:, :, None]])
  # Define output layer
  _output = LinearCell(FLAGS.n_output_embed, FLAGS.n_class, prefix='output')
  tparams = merge_dict(tparams, _output._params)
  # Call output layer
  h_logit = _output([h_o_embed])
  logit_shape = h_logit.shape
  logit = h_logit.reshape([logit_shape[0]*logit_shape[1], logit_shape[2]])
  logit = logit - logit.max(axis=1).dimshuffle(0, 'x')
  probs = logit - tensor.log(tensor.exp(logit).sum(axis=1).dimshuffle(0, 'x'))
  # Compute the cost
  y_flat = y.flatten()
  y_flat_idx = tensor.arange(y_flat.shape[0]) * FLAGS.n_class + y_flat
  cost = -probs.flatten()[y_flat_idx]
  cost = cost.reshape([y.shape[0], y.shape[1]])
  cost = (cost * y_mask).sum(0)
  cost_len = y_mask.sum(0)
  last_state = tensor.stack([last_state_1, last_state_2, last_state_3], axis=0)
  f_prop_updates = OrderedDict()
  f_prop_updates[tstate] = last_state
  states = [tstate]
  # Later use for visualization
  inps = [inp, inp_mask]
  print("Building f_log_prob function")
  f_log_prob = theano.function(inps, [cost, cost_len], updates=f_prop_updates)
  cost = cost.mean()
  # If the flag is on, apply L2 regularization on weights
  if FLAGS.weight_decay > 0.:
    weights_norm = 0.
    for k, v in tparams.iteritems():
      weights_norm += (v**2).sum()
    cost += weights_norm * FLAGS.weight_decay
  #print("Computing the gradients")
  grads = tensor.grad(cost, wrt=itemlist(tparams))
  grads = gradient_clipping(grads, tparams, 1.)
  # Compile the optimizer, the actual computational graph
  learning_rate = tensor.scalar(name='learning_rate')
  gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
             for k, p in tparams.iteritems()]
  gsup = OrderedDict(izip(gshared, grads))
  print("Building f_prop function")
  f_prop = theano.function(inps, [cost],
                           updates=merge_dict(gsup, f_prop_updates))
  opt_updates, opt_tparams = adam(learning_rate, tparams, gshared)
  if FLAGS.start_from_ckpt and os.path.exists(opt_file_name):
    opt_params = np.load(opt_file_name)
    zipp(opt_params, opt_tparams)
  print("Building f_update function")
  f_update = theano.function([learning_rate], [], updates=opt_updates,
                             on_unused_input='ignore')
  #print("Building f_debug function")
  f_debug = theano.function(inps,
                            [h_rnn_1_3d, h_rnn_2_3d, h_rnn_3_3d],
                            updates=f_prop_updates,
                            on_unused_input='ignore')
  return f_prop, f_update, f_log_prob, f_debug, tparams, opt_tparams, states, None


def main(FLAGS):
  if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
  train = TrainModel(FLAGS, build_graph, monitor)
  train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-learning_rate', type=float, default=0.002)
  parser.add_argument('-weight_decay', type=float, default=0.0005)
  parser.add_argument('-batch_size', type=int, default=64)
  parser.add_argument('-n_epoch', type=int, default=500)
  parser.add_argument('-display_freq', type=int, default=100)
  parser.add_argument('-max_seq_len', type=int, default=100)
  parser.add_argument('-n_hidden', type=int, default=512)
  parser.add_argument('-n_class', type=int, default=50)
  parser.add_argument('-n_input_embed', type=int, default=128)
  parser.add_argument('-n_output_embed', type=int, default=512)
  parser.add_argument('-base_seed', type=int, default=20170309)
  parser.add_argument('-add_seed', type=int, default=0)
  parser.add_argument('-start_from_ckpt', action="store_true", default=False)
  parser.add_argument('-eval_train', action="store_true", default=False)
  parser.add_argument('-use_slope_anneal', action='store_true', default=False)
  parser.add_argument('-n_slope', type=int, default=5)
  parser.add_argument('-n_anneal_epoch', type=int, default=100)
  parser.add_argument('-log_dir', type=str)
  FLAGS = parser.parse_args()
  main(FLAGS)
