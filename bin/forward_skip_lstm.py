#!/usr/bin/env python

import os
import sys
sys.path.insert(0, '..')

from itertools import islice

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple
from mixer import gen_mask
from mixer import insert_item2dict
from mixer import save_npz2
from mixer import skip_rnn_act, skip_rnn_act_parallel, aggr_skip_rnn_act_parallel
from mixer import LinearVF, compute_advantage
from mixer import categorical_ent, expand_pred_idx
from model import LinearCell
from model import LSTMModule

from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data, StopWatch

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch-size', 64, 'Size of mini-batch')
flags.DEFINE_integer('n-epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('n-input', 123, 'Number of RNN hidden units')
flags.DEFINE_integer('n-hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n-class', 3436, 'Number of target symbols')
flags.DEFINE_integer('n-action', 3, 'Number of actions (max skim size)')
flags.DEFINE_integer('n-fast-action', 10, 'Number of steps to skip in the fast action mode')
flags.DEFINE_boolean('fast-action', False, 'If true, operate in the fast action mode')
flags.DEFINE_boolean('ref-input', False, 'If true, policy refers input')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('metafile', 'skip_lstm_wsj', 'Directory path to files')
flags.DEFINE_boolean('no-copy', True, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data-path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('dataset', 'test_dev93', '')
flags.DEFINE_float('discount-gamma', 0.99, 'discount_factor')
flags.DEFINE_string('wxfilename', 'ark:-', '')
flags.DEFINE_string('metafile', 'best_model.ckpt-1000.meta', '')

tg_fields = ['ml_cost',
             'rl_cost',
             'rl_ent_cost',
             'seq_x_data',
             'seq_x_mask',
             'seq_y_data',
             'init_state',
             'seq_action', 
             'seq_advantage',
             'seq_action_mask',
             'pred_idx']

sg_fields = ['step_h_state',
             'step_last_state',
             'step_label_probs',
             'step_action_probs',
             'step_action_samples',
             'step_x_data',
             'prev_states',
             'action_entropy']

TrainGraph = namedtuple('TrainGraph', ' '.join(tg_fields))
SampleGraph = namedtuple('SampleGraph', ' '.join(sg_fields))

def build_graph(args):
  with tf.device(args.device):
    # [batch_size, seq_len, ...]
    seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input))
    seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
    seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

    # [2, batch_size, ...]
    init_state = tf.placeholder(tf.float32, shape=(2, None, args.n_hidden))

    seq_action = tf.placeholder(tf.float32, shape=(None, None, args.n_action))
    seq_advantage = tf.placeholder(tf.float32, shape=(None, None))
    seq_action_mask = tf.placeholder(tf.float32, shape=(None, None))
    
    # input data (batch_size, feat_size) for sampling
    step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input))
    # previous state (2, batch_size, num_hiddens)
    prev_states = tf.placeholder(tf.float32, shape=(2, None, args.n_hidden))

  with tf.variable_scope('rnn'):
    _rnn = LSTMModule(num_units=args.n_hidden)

  with tf.variable_scope('label'):
    _label_logit = LinearCell(num_units=args.n_class)

  with tf.variable_scope('action'):
    _action_logit = LinearCell(num_units=args.n_action)

  # sampling graph
  step_h_state, step_last_state = _rnn(step_x_data, prev_states, one_step=True)

  step_label_logits = _label_logit(step_h_state, 'label_logit')
  step_label_probs = tf.nn.softmax(logits=step_label_logits)

  if FLAGS.ref_input:
    step_action_logits = _action_logit([step_x_data, step_h_state], 'action_logit')
  else:
    step_action_logits = _action_logit(step_h_state, 'action_logit')
  step_action_probs = tf.nn.softmax(logits=step_action_logits)
  step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1)
  step_action_entropy = categorical_ent(step_action_probs)

  # training graph
  seq_hid_3d, _ = _rnn(seq_x_data, init_state)
  seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

  seq_label_logits = _label_logit(seq_hid_2d, 'label_logit')

  y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=FLAGS.n_class)

  ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
    labels=y_1hot)
  ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

  pred_idx = tf.argmax(seq_label_logits, axis=1)

  seq_hid_3d_rl = seq_hid_3d[:,:-1,:]
  seq_hid_2d_rl = tf.reshape(seq_hid_3d_rl, [-1, args.n_hidden])
  seq_hid_2d_rl = tf.stop_gradient(seq_hid_2d_rl)

  if FLAGS.ref_input:
    seq_action_logits = _action_logit([tf.reshape(seq_x_data[:, :-1, :], [-1, args.n_input]), seq_hid_2d_rl], 'action_logit')
  else:
    seq_action_logits = _action_logit(seq_hid_2d_rl, 'action_logit')

  seq_action_probs = tf.nn.softmax(seq_action_logits)

  action_prob_entropy = categorical_ent(seq_action_probs)
  action_prob_entropy *= tf.reshape(seq_action_mask, [-1])
  action_prob_entropy = tf.reduce_sum(action_prob_entropy)/tf.reduce_sum(seq_action_mask)

  rl_cost = tf.reduce_sum(tf.log(seq_action_probs+1e-8) \
    * tf.reshape(seq_action, [-1,args.n_action]), axis=-1)
  rl_cost *= tf.reshape(seq_advantage, [-1])
  rl_cost = -tf.reduce_sum(rl_cost*tf.reshape(seq_action_mask, [-1]))

  rl_ent_cost = -action_prob_entropy

  train_graph = TrainGraph(ml_cost,
                           rl_cost,
                           rl_ent_cost,
                           seq_x_data,
                           seq_x_mask,
                           seq_y_data,
                           init_state,
                           seq_action,
                           seq_advantage,
                           seq_action_mask, 
                           pred_idx)

  sample_graph = SampleGraph(step_h_state,
                             step_last_state,
                             step_label_probs,
                             step_action_probs,
                             step_action_samples,
                             step_x_data,
                             prev_states,
                             step_action_entropy)

  return train_graph, sample_graph


def initial_states(batch_size, n_hidden):
  init_state = np.zeros([2, batch_size, n_hidden], dtype=np.float32)
  return init_state

def run_skip_rnn(x, x_mask, sess, sg, args):
  pass

def main(_):
  print(' '.join(sys.argv), file=sys.stderr)
  args = FLAGS
  print(args.__flags, file=sys.stderr)

  tf.get_variable_scope()._reuse = None

  meta_file = args.metafile
  model_file = meta_file[:-5]

  tg, sg = build_graph(args)

  global_step = tf.Variable(0, trainable=False, name="global_step")

  sync_data(args)
  test_set = create_ivector_test_datastream(args.data_path, args.dataset, args.batch_size)

  uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.batch_size) 

  save_op = tf.train.Saver(max_to_keep=5)

  gen_episodes = aggr_skip_rnn_act_parallel

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print('Loading model...', file=sys.stderr)
    save_op = tf.train.import_meta_graph(meta_file)
    save_op.restore(sess, model_file)

    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    print('Computing label probs...', file=sys.stderr)

    for bidx, (batch, uttid_batch) in enumerate(zip(test_set.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
      orig_x, orig_x_mask, _, _ = batch
      n_batch, n_seq, _ = x.shape
      uttid_batch, = uttid_batch

      feat_lens = orig_x_mask.sum(axis=1)

      feed_states = initial_states(n_batch, args.n_hidden)
      seq_label_probs = run_skip_rnn(orig_x, orig_x_mask, sess, sg, args)

      seq_label_probs = expand_pred_idx(actions, orig_x_mask, seq_label_probs, n_batch, args)

      for out_idx, (output, uttid) in enumerate(zip(seq_label_probs, uttid_batch)):
        valid_len = int(feat_lens[out_idx])
        writer.write(uttid.encode('ascii'), np.log(output[:valid_len]))

      print('.', file=sys.stderr, end='')

    print('', file=sys.stderr)
    print('Done', file=sys.stderr)
      
if __name__ == '__main__':
  tf.app.run()




