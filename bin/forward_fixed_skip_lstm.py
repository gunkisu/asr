#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf

import glob

from skiprnn.model import LinearCell
from skiprnn.model import LSTMModule
from skiprnn.mixer import save_npz2

from collections import namedtuple, OrderedDict

from data.fuel_utils import create_ivector_datastream, get_uttid_stream
from libs.utils import sync_data, skip_frames_fixed, StopWatch

from itertools import islice

import kaldi_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning-rate', 0.002, 'Initial learning rate')
flags.DEFINE_integer('batch-size', 64, 'Size of mini-batch')
flags.DEFINE_integer('min-after-cache', 1024, 'Size of mini-batch')
flags.DEFINE_integer('n-input', 123, 'Number of RNN hidden units')
flags.DEFINE_integer('n-hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n-class', 3436, 'Number of target symbols')
flags.DEFINE_integer('base-seed', 20170309, 'Base random seed') 
flags.DEFINE_integer('add-seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('log-dir', 'skip_lstm_wsj', 'Directory path to files')
flags.DEFINE_boolean('no-copy', True, '') # test set is typically small
flags.DEFINE_boolean('no-length-sort', False, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data-path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('dataset', 'test_dev93', '')
flags.DEFINE_integer('n-skip', 1, 'Number of frames to skip')
flags.DEFINE_string('wxfilename', 'ark:-', '')

TrainGraph = namedtuple('TrainGraph', 'ml_cost seq_x_data seq_x_mask seq_y_data init_state pred_idx seq_label_probs')

def build_graph(args):
  with tf.device(args.device):
    # [batch_size, seq_len, ...]
    seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input))
    seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
    seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

    # [2, batch_size, ...]
    init_state = tf.placeholder(tf.float32, shape=(2, None, args.n_hidden))

  with tf.variable_scope('rnn'):
    _rnn = LSTMModule(num_units=args.n_hidden)

  with tf.variable_scope('label'):
    _label_logit = LinearCell(num_units=args.n_class)

  # training graph
  seq_hid_3d, _ = _rnn(seq_x_data, init_state)
  seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

  seq_label_logits = _label_logit(seq_hid_2d, 'label_logit')

  y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=FLAGS.n_class)

  ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
    labels=y_1hot)
  ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

  pred_idx = tf.argmax(seq_label_logits, axis=1)

  seq_label_probs = tf.nn.softmax(seq_label_logits)

  train_graph = TrainGraph(ml_cost,
                           seq_x_data,
                           seq_x_mask,
                           seq_y_data,
                           init_state,
                           pred_idx,
                           seq_label_probs)

  return train_graph

def initial_states(batch_size, n_hidden):
  init_state = np.zeros([2, batch_size, n_hidden], dtype=np.float32)
  return init_state

def main(_):
  print(' '.join(sys.argv), file=sys.stderr)
  args = FLAGS
  print(args.__flags, file=sys.stderr)

  tf.get_variable_scope()._reuse = None

  _seed = args.base_seed + args.add_seed
  tf.set_random_seed(_seed)
  np.random.seed(_seed)

  meta_filename, = glob.glob(os.path.join(args.log_dir, '*best*.meta'))
  meta_file = os.path.basename(meta_filename)
  model_file = meta_file[:-5]

  tg = build_graph(args)

  global_step = tf.Variable(0, trainable=False, name="global_step")

  sync_data(args)
  test_set = create_ivector_datastream(path=args.data_path, which_set=args.dataset, 
    batch_size=args.batch_size, min_after_cache=args.min_after_cache, 
    length_sort=not args.no_length_sort)

  uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.batch_size) 

  save_op = tf.train.Saver(max_to_keep=5)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        
    print('Loading model...', file=sys.stderr)
    save_op = tf.train.import_meta_graph(os.path.join(args.log_dir, meta_file))
    save_op.restore(sess, os.path.join(args.log_dir, model_file))

    writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

    print('Computing label probs...', file=sys.stderr)

    for bidx, (batch, uttid_batch) in enumerate(zip(test_set.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
      orig_x, orig_x_mask, _, _, orig_y, _ = batch
      uttid_batch, = uttid_batch

      feat_lens = orig_x_mask.sum(axis=1)

      sub_batch, = list(skip_frames_fixed([orig_x, orig_x_mask, orig_y], args.n_skip+1, return_first=True))
      x, x_mask, y = sub_batch
      n_batch, n_seq, _ = x.shape

      feed_states = initial_states(n_batch, args.n_hidden)

      seq_label_probs, = sess.run([tg.seq_label_probs],
        feed_dict={tg.seq_x_data: x, tg.seq_x_mask: x_mask,
        tg.seq_y_data: y, tg.init_state: feed_states})

      seq_label_probs = seq_label_probs.reshape([n_batch, n_seq, -1])
      seq_label_probs = seq_label_probs.repeat(args.n_skip+1, axis=1)

      for out_idx, (output, uttid) in enumerate(zip(seq_label_probs, uttid_batch)):
        valid_len = int(feat_lens[out_idx])
        writer.write(uttid.encode('ascii'), np.log(output[:valid_len]))

      print('.', file=sys.stderr, end='')

    print('', file=sys.stderr)
    print('Done', file=sys.stderr)

if __name__ == '__main__':
  tf.app.run()

