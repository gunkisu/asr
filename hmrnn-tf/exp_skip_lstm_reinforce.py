#!/usr/bin/env python

import os
import sys
sys.path.insert(0, '..')
import itertools

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple
from mixer import gen_mask
from mixer import insert_item2dict
from mixer import save_npz2
from mixer import skip_rnn_act, skip_rnn_act_parallel
from mixer import LinearVF, compute_advantage
from mixer import categorical_ent
from model import LinearCell
from model import LSTMModule

from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data, StopWatch

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate')
flags.DEFINE_float('rl_learning_rate', 0.01, 'Initial learning rate for RL')
flags.DEFINE_integer('batch_size', 64, 'Size of mini-batch')
flags.DEFINE_integer('n_epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('display_freq', 100, 'Display frequency')
flags.DEFINE_integer('n_input', 123, 'Number of RNN hidden units')
flags.DEFINE_integer('n_hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n_class', 3436, 'Number of target symbols')
flags.DEFINE_integer('n_action', 3, 'Number of actions (max skim size)')
flags.DEFINE_integer('base_seed', 20170309, 'Base random seed') 
flags.DEFINE_integer('add_seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start_from_ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad_clip', True, 'If true, clip the gradients')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('log_dir', 'skip_lstm_wsj', 'Directory path to files')
flags.DEFINE_boolean('no_copy', False, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data_path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('train_dataset', 'train_si284', '')
flags.DEFINE_string('valid_dataset', 'test_dev93', '')
flags.DEFINE_string('test_dataset', 'test_eval92', '')
flags.DEFINE_float('discount_gamma', 0.99, 'discount_factor')

TrainGraph = namedtuple('TrainGraph', 'ml_cost rl_cost seq_x_data seq_x_mask seq_y_data init_state seq_action seq_advantage, seq_action_mask')
SampleGraph = namedtuple('SampleGraph', 'step_h_state step_last_state step_label_probs step_action_probs step_x_data prev_states action_entropy')

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

  step_action_logits = _action_logit(step_h_state, 'action_logit')
  step_action_probs = tf.nn.softmax(logits=step_action_logits)

  # [batch_size]
  action_entropy = categorical_ent(step_action_probs)

  # training graph
  seq_hid_3d, _ = _rnn(seq_x_data, init_state)
  seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

  seq_label_logits = _label_logit(seq_hid_2d, 'label_logit')

  y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=FLAGS.n_class)

  ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
    labels=y_1hot)
  ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

  seq_hid_3d_rl = seq_hid_3d[:,:-1,:]
  seq_hid_2d_rl = tf.reshape(seq_hid_3d_rl, [-1, args.n_hidden])

  seq_action_logits = _action_logit(seq_hid_2d_rl, 'action_logit')
  seq_action_probs = tf.nn.softmax(seq_action_logits)

  rl_cost = tf.reduce_sum(tf.log(seq_action_probs+1e-8) \
    * tf.reshape(seq_action, [-1,args.n_action]), axis=-1)
  rl_cost *= tf.reshape(seq_advantage, [-1])
  rl_cost = tf.reduce_sum(rl_cost*tf.reshape(seq_action_mask, [-1]))

  train_graph = TrainGraph(ml_cost,
                           rl_cost,
                           seq_x_data,
                           seq_x_mask,
                           seq_y_data,
                           init_state,
                           seq_action,
                           seq_advantage,
                           seq_action_mask)

  sample_graph = SampleGraph(step_h_state,
                             step_last_state,
                             step_label_probs,
                             step_action_probs,
                             step_x_data,
                             prev_states,
                             action_entropy)

  return train_graph, sample_graph


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

  tg, sg = build_graph(args)
  tg_ml_cost = tf.reduce_mean(tg.ml_cost)

  global_step = tf.Variable(0, trainable=False, name="global_step")

  tvars = tf.trainable_variables()
  ml_vars = [tvar for tvar in tvars if "action_logit" not in tvar.name]
  rl_vars = [tvar for tvar in tvars if "action_logit" in tvar.name]

  ml_opt_func = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                       beta1=0.9, beta2=0.99)
  rl_opt_func = tf.train.AdamOptimizer(learning_rate=args.rl_learning_rate,
                                       beta1=0.9, beta2=0.99)

  if args.grad_clip:
    ml_grads, _ = tf.clip_by_global_norm(tf.gradients(tg_ml_cost, ml_vars),
                                      clip_norm=1.0)
  else:
    ml_grads = tf.gradients(tg_ml_cost, ml_vars)
  ml_op = ml_opt_func.apply_gradients(zip(ml_grads, ml_vars), global_step=global_step)

  tg_rl_cost = tf.reduce_mean(tg.rl_cost)
  rl_grads = tf.gradients(tg_rl_cost, rl_vars)
  rl_op = rl_opt_func.apply_gradients(zip(rl_grads, rl_vars), global_step=global_step)

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

  vf = LinearVF()

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
    tr_rl_costs = []
    tr_action_entropies = []
    tr_rewards = []

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

      print('Epoch {} training'.format(_epoch+1))
      
      # For each batch 
      for batch in train_set.get_epoch_iterator():
        x, x_mask, _, _, y, _ = batch
        x = np.transpose(x, (1, 0, 2))
        x_mask = np.transpose(x_mask, (1, 0))
        y = np.transpose(y, (1, 0))
        _, n_batch, _ = x.shape
        _n_exp += n_batch

        new_x, new_y, actions, rewards, action_entropies, new_x_mask, new_reward_mask = \
            skip_rnn_act(x, x_mask, y, sess, sg, args)

        advantages = compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args)
                  
        _feed_states = initial_states(n_batch, args.n_hidden)

        _tr_ml_cost, _tr_rl_cost, _, _ = \
          sess.run([tg.ml_cost, tg.rl_cost, ml_op, rl_op],
                 feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask,
                      tg.seq_y_data: new_y, tg.init_state: _feed_states,
                      tg.seq_action: actions, tg.seq_advantage: advantages, 
                      tg.seq_action_mask: new_reward_mask})

        _tr_ce = _tr_ml_cost.sum() / new_x_mask.sum()
        _tr_ce_summary, = sess.run([tr_ce_summary], feed_dict={tr_ce: _tr_ce})
        summary_writer.add_summary(_tr_ce_summary, global_step.eval())

        tr_ces.append(_tr_ce)
        tr_rl_costs.append(_tr_rl_cost.mean())
        tr_action_entropies.append(action_entropies.mean())
        tr_rewards.append(rewards.sum())
                  
        if global_step.eval() % args.display_freq == 0:
          avg_tr_ce = np.asarray(tr_ces).mean()
          avg_tr_rl_cost = np.asarray(tr_rl_costs).mean()
          avg_tr_action_entropy = np.asarray(tr_action_entropies).mean()
          avg_tr_reward = np.asarray(tr_rewards).mean()

          print("TRAIN: epoch={} iter={} ml_cost(ce/frame)={:.2f} rl_cost={:.2f} reward={:.2f} action_entropy={:.2f} time_taken={:.2f}".format(
              _epoch, global_step.eval(), avg_tr_ce, avg_tr_rl_cost, avg_tr_reward, avg_tr_action_entropy, disp_sw.elapsed()))

          tr_ces = []
          tr_rl_costs = []
          tr_action_entropies = []
          tr_rewards = []
          
          disp_sw.reset()

      print('--')
      print('End of epoch {}'.format(_epoch+1))
      epoch_sw.print_elapsed()

      print('Testing')

      # Evaluate the model on the validation set
      val_ces = []
      val_rl_costs = []
      val_action_entropies = []
      val_rewards = []

      eval_sw.reset()
      for batch in valid_set.get_epoch_iterator():
        x, x_mask, _, _, y, _ = batch
        x = np.transpose(x, (1, 0, 2))
        x_mask = np.transpose(x_mask, (1, 0))
        y = np.transpose(y, (1, 0))
        _, n_batch, _ = x.shape

        new_x, new_y, actions, rewards, action_entropies, new_x_mask, new_reward_mask = \
            skip_rnn_act(x, x_mask, y, sess, sg, args)
        advantages = compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args)

        _feed_states = initial_states(n_batch, args.n_hidden)

        _val_ml_cost, _val_rl_cost = sess.run([tg.ml_cost, tg.rl_cost, ],
                feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask,
                      tg.seq_y_data: new_y, tg.init_state: _feed_states,
                      tg.seq_action: actions, tg.seq_advantage: advantages, 
                      tg.seq_action_mask: new_reward_mask})
        
        _val_ce = _val_ml_cost.sum() / new_x_mask.sum()

        val_ces.append(_val_ce)
        val_rl_costs.append(_val_rl_cost.mean())
        val_action_entropies.append(action_entropies.mean())
        val_rewards.append(rewards.sum())

      avg_val_ce = np.asarray(val_ces).mean()
      avg_val_rl_cost = np.asarray(val_rl_costs).mean()
      avg_val_action_entropy = np.asarray(val_action_entropies).mean()
      avg_val_reward = np.asarray(val_rewards).mean()

      print("VALID: epoch={} ml_cost(ce/frame)={:.2f} rl_cost={:.2f} reward={:.2f} action_entropy={:.2f} time_taken={:.2f}".format(
          _epoch, avg_val_ce, avg_val_rl_cost, avg_val_reward, avg_val_action_entropy, eval_sw.elapsed()))

      _val_ce_summary, = sess.run([val_ce_summary], feed_dict={val_ce: avg_val_ce}) 
      summary_writer.add_summary(_val_ce_summary, global_step.eval())
      
      insert_item2dict(eval_summary, 'val_ce', avg_val_ce)
      insert_item2dict(eval_summary, 'val_rl_cost', avg_val_rl_cost)
      insert_item2dict(eval_summary, 'val_reward', avg_val_reward)
      insert_item2dict(eval_summary, 'val_action_entropy', avg_val_action_entropy)
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




