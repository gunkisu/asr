#!/usr/bin/env python
import sys
import os
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import nats2bits
from mixer import insert_item2dict
from model import LinearCell
from model import LSTMModule
from mixer import save_npz2

from collections import namedtuple, OrderedDict

from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data, skip_frames_fixed, StopWatch

from itertools import islice

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning-rate', 0.002, 'Initial learning rate')
flags.DEFINE_integer('batch-size', 64, 'Size of mini-batch')
flags.DEFINE_integer('n-epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('display-freq', 100, 'Display frequency')
flags.DEFINE_integer('n-input', 123, 'Number of RNN hidden units')
flags.DEFINE_integer('n-hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n-class', 3436, 'Number of target symbols')
flags.DEFINE_integer('base-seed', 20170309, 'Base random seed') 
flags.DEFINE_integer('add-seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start-from-ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad-clip', True, 'If true, clip the gradients')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('log-dir', 'skip_lstm_wsj', 'Directory path to files')
flags.DEFINE_boolean('no-copy', False, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data-path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('train-dataset', 'train_si284', '')
flags.DEFINE_string('valid-dataset', 'test_dev93', '')
flags.DEFINE_string('test-dataset', 'test_eval92', '')
flags.DEFINE_integer('n-skip', 1, 'Number of frames to skip')

TrainGraph = namedtuple('TrainGraph', 'ml_cost seq_x_data seq_x_mask seq_y_data init_state, pred_idx')

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

  train_graph = TrainGraph(ml_cost,
                           seq_x_data,
                           seq_x_mask,
                           seq_y_data,
                           init_state,
                           pred_idx)

  return train_graph

def initial_states(batch_size, n_hidden):
  init_state = np.zeros([2, batch_size, n_hidden], dtype=np.float32)
  return init_state

def main(_):
  print(' '.join(sys.argv))
  args = FLAGS
  print(args.__flags)
  if not args.start_from_ckpt:
    if tf.gfile.Exists(args.log_dir):
      tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)

  tf.get_variable_scope()._reuse = None

  _seed = args.base_seed + args.add_seed
  tf.set_random_seed(_seed)
  np.random.seed(_seed)

  prefix_name = os.path.join(args.log_dir, 'model')
  file_name = '%s.npz' % prefix_name

  eval_summary = OrderedDict() # 

  tg = build_graph(args)
  tg_ml_cost = tf.reduce_mean(tg.ml_cost)

  global_step = tf.Variable(0, trainable=False, name="global_step")

  tvars = tf.trainable_variables()

  ml_opt_func = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                       beta1=0.9, beta2=0.99)

  if args.grad_clip:
    ml_grads, _ = tf.clip_by_global_norm(tf.gradients(tg_ml_cost, tvars),
                                      clip_norm=1.0)
  else:
    ml_grads = tf.gradients(tg_ml_cost, tvars)
  ml_op = ml_opt_func.apply_gradients(zip(ml_grads, tvars), global_step=global_step)

  sync_data(args)
  datasets = [args.train_dataset, args.valid_dataset, args.test_dataset]
  train_set, valid_set, test_set = [create_ivector_datastream(path=args.data_path, which_set=dataset, 
      batch_size=args.batch_size) for dataset in datasets]

  init_op = tf.global_variables_initializer()
  save_op = tf.train.Saver(max_to_keep=5)
  best_save_op = tf.train.Saver(max_to_keep=5)

  with tf.name_scope("per_step_eval"):
    tr_ce = tf.placeholder(tf.float32)
    tr_ce_summary = tf.summary.scalar("tr_ce", tr_ce)

  with tf.name_scope("per_epoch_eval"):
    best_val_ce = tf.placeholder(tf.float32)
    val_ce = tf.placeholder(tf.float32)
    best_val_ce_summary = tf.summary.scalar("best_valid_ce", best_val_ce)
    val_ce_summary = tf.summary.scalar("valid_ce", val_ce)

  with tf.Session() as sess:
    sess.run(init_op)

    if args.start_from_ckpt:
      save_op = tf.train.import_meta_graph(os.path.join(args.log_dir,
                                                        'model.ckpt.meta'))
      save_op.restore(sess, os.path.join(args.log_dir, 'model.ckpt'))
      print("Restore from the last checkpoint. "
            "Restarting from %d step." % global_step.eval())

    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph, flush_secs=5.0)

    tr_ces = []
    tr_acc_sum = 0
    tr_acc_count = 0
    _best_score = np.iinfo(np.int32).max

    epoch_sw = StopWatch()
    disp_sw = StopWatch()
    eval_sw = StopWatch()
    per_sw = StopWatch()
    # For each epoch 
    for _epoch in xrange(args.n_epoch):
      _n_exp = 0

      epoch_sw.reset()
      disp_sw.reset()

      print('--')
      print('Epoch {} training'.format(_epoch+1))
      
      # For each batch 
      for batch in train_set.get_epoch_iterator():
        orig_x, orig_x_mask, _, _, orig_y, _ = batch
         
        for sub_batch in skip_frames_fixed([orig_x, orig_x_mask, orig_y], args.n_skip+1):
            x, x_mask, y = sub_batch
            n_batch, _, _ = x.shape
            _n_exp += n_batch

            _feed_states = initial_states(n_batch, args.n_hidden)

            _tr_ml_cost, _pred_idx, _ = sess.run([tg.ml_cost, tg.pred_idx, ml_op],
                     feed_dict={tg.seq_x_data: x, tg.seq_x_mask: x_mask,
                          tg.seq_y_data: y, tg.init_state: _feed_states})

            _tr_ce = _tr_ml_cost.sum() / x_mask.sum()
            _tr_ce_summary, = sess.run([tr_ce_summary], feed_dict={tr_ce: _tr_ce})
            summary_writer.add_summary(_tr_ce_summary, global_step.eval())

            _, n_seq = orig_y.shape
            _pred_idx = _pred_idx.reshape([n_batch, -1]).repeat(args.n_skip+1, axis=1)
            _pred_idx = _pred_idx[:,:n_seq]

            tr_ces.append(_tr_ce)
            tr_acc_sum += ((_pred_idx == orig_y) * orig_y).sum()
            tr_acc_count += orig_y.sum()
                 
        if global_step.eval() % args.display_freq == 0:
          avg_tr_ce = np.asarray(tr_ces).mean()
          avg_tr_fer = 1. - float(tr_acc_sum) / tr_acc_count

          print("TRAIN: epoch={} iter={} ml_cost(ce/frame)={:.2f} fer={:.2f} time_taken={:.2f}".format(
              _epoch, global_step.eval(), avg_tr_ce, avg_tr_fer, disp_sw.elapsed()))

          tr_ces = []
          tr_acc_sum = 0
          tr_acc_count = 0
          disp_sw.reset()

      print('--')
      print('End of epoch {}'.format(_epoch+1))
      epoch_sw.print_elapsed()

      print('Testing')

      # Evaluate the model on the validation set
      val_ces = []
      val_acc_sum = 0
      val_acc_count = 0

      eval_sw.reset()
      for batch in valid_set.get_epoch_iterator():
        orig_x, orig_x_mask, _, _, orig_y, _ = batch
         
        for sub_batch in skip_frames_fixed([orig_x, orig_x_mask, orig_y], args.n_skip+1, return_first=True):
            x, x_mask, y = sub_batch
            n_batch, _, _ = x.shape

            _feed_states = initial_states(n_batch, args.n_hidden)

            _val_ml_cost, _pred_idx = sess.run([tg.ml_cost, tg.pred_idx],
                    feed_dict={tg.seq_x_data: x, tg.seq_x_mask: x_mask,
                          tg.seq_y_data: y, tg.init_state: _feed_states})
            
            _val_ce = _val_ml_cost.sum() / x_mask.sum()

            _, n_seq = orig_y.shape
            _pred_idx = _pred_idx.reshape([n_batch, -1]).repeat(args.n_skip+1, axis=1)
            _pred_idx = _pred_idx[:,:n_seq]

            val_acc_sum = ((_pred_idx == orig_y) * orig_y).sum()
            val_acc_count += orig_y.sum()

            val_ces.append(_val_ce)
        
      avg_val_ce = np.asarray(val_ces).mean()
      avg_val_fer = 1. - float(val_acc_sum) / val_acc_count

      print("VALID: epoch={} ml_cost(ce/frame)={:.2f} fer={:.2f} time_taken={:.2f}".format(
          _epoch, avg_val_ce, avg_val_fer, eval_sw.elapsed()))

      _val_ce_summary, = sess.run([val_ce_summary], feed_dict={val_ce: avg_val_ce}) 
      summary_writer.add_summary(_val_ce_summary, global_step.eval())
      
      insert_item2dict(eval_summary, 'val_ce', avg_val_ce)
      insert_item2dict(eval_summary, 'time', eval_sw.elapsed())
      save_npz2(file_name, eval_summary)

      # Save model
      if avg_val_ce < _best_score:
        _best_score = avg_val_ce
        best_ckpt = best_save_op.save(sess, os.path.join(args.log_dir,
                                                         "best_model.ckpt"),
                                      global_step=global_step)
        print("Best checkpoint stored in: %s" % best_ckpt)
      ckpt = save_op.save(sess, os.path.join(args.log_dir, "model.ckpt"),
                          global_step=global_step)
      print("Checkpoint stored in: %s" % ckpt)

      _best_val_ce_summary, = sess.run([best_val_ce_summary],
                                feed_dict={best_val_ce: _best_score})
      summary_writer.add_summary(_best_val_ce_summary, global_step.eval())
    summary_writer.close()

    print("Optimization Finished.")

if __name__ == '__main__':
  tf.app.run()

