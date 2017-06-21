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
flags.DEFINE_float('learning-rate', 0.002, 'Initial learning rate')
flags.DEFINE_float('rl-learning-rate', 0.01, 'Initial learning rate for RL')
flags.DEFINE_integer('min-after-cache', 1024, 'Size of mini-batch')
flags.DEFINE_float('ent-weight', 0.1, 'entropy regularizer weight')
flags.DEFINE_integer('batch-size', 64, 'Size of mini-batch')
flags.DEFINE_integer('n-epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('display-freq', 100, 'Display frequency')
flags.DEFINE_integer('n-input', 123, 'Number of RNN hidden units')
flags.DEFINE_integer('n-hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n-class', 3436, 'Number of target symbols')
flags.DEFINE_integer('n-action', 3, 'Number of actions (max skim size)')
flags.DEFINE_integer('n-fast-action', 10, 'Number of steps to skip in the fast action mode')
flags.DEFINE_integer('base-seed', 20170309, 'Base random seed') 
flags.DEFINE_integer('add-seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start-from-ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad-clip', True, 'If true, clip the gradients')
flags.DEFINE_boolean('parallel', True, 'If true, do parallel sampling')
flags.DEFINE_boolean('aggr-reward', True, 'If true, use reward from FER within skimm')
flags.DEFINE_boolean('fast-action', False, 'If true, operate in the fast action mode')
flags.DEFINE_boolean('ref-input', False, 'If true, policy refers input')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('log-dir', 'skip_lstm_wsj', 'Directory path to files')
flags.DEFINE_boolean('no-copy', False, '')
flags.DEFINE_boolean('no-length-sort', False, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data-path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('train-dataset', 'train_si284', '')
flags.DEFINE_string('valid-dataset', 'test_dev93', '')
flags.DEFINE_string('test-dataset', 'test_eval92', '')
flags.DEFINE_float('discount-gamma', 0.99, 'discount_factor')

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
    step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')
    # previous state (2, batch_size, num_hiddens)
    prev_states = tf.placeholder(tf.float32, shape=(2, None, args.n_hidden), name='prev_states')

  with tf.variable_scope('rnn'):
    _rnn = LSTMModule(num_units=args.n_hidden)

  with tf.variable_scope('label'):
    _label_logit = LinearCell(num_units=args.n_class)

  with tf.variable_scope('action'):
    _action_logit = LinearCell(num_units=args.n_action)

  # sampling graph
  step_h_state, step_last_state = _rnn(step_x_data, prev_states, one_step=True)

  step_label_logits = _label_logit(step_h_state, 'label_logit')
  step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

  if FLAGS.ref_input:
    step_action_logits = _action_logit([step_x_data, step_h_state], 'action_logit')
  else:
    step_action_logits = _action_logit(step_h_state, 'action_logit')
  step_action_probs = tf.nn.softmax(logits=step_action_logits, name='step_action_probs')
  step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1, name='step_action_samples')
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

  eval_summary = OrderedDict()

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

  tg_rl_cost = tf.reduce_mean(tg.rl_cost) + tg.rl_ent_cost*args.ent_weight
  rl_grads = tf.gradients(tg_rl_cost, rl_vars)
  rl_op = rl_opt_func.apply_gradients(zip(rl_grads, rl_vars), global_step=global_step)

  sync_data(args)
  datasets = [args.train_dataset, args.valid_dataset, args.test_dataset]
  train_set, valid_set, test_set = [create_ivector_datastream(path=args.data_path, which_set=dataset, 
      batch_size=args.batch_size, min_after_cache=args.min_after_cache, length_sort=not args.no_length_sort) for dataset in datasets]

  init_op = tf.global_variables_initializer()
  save_op = tf.train.Saver(max_to_keep=5)
  best_save_op = tf.train.Saver(max_to_keep=5)

  with tf.name_scope("per_step_eval"):
    tr_ce = tf.placeholder(tf.float32)
    tr_ce_summary = tf.summary.scalar("tr_ce", tr_ce)
    tr_image = tf.placeholder(tf.float32)
    tr_image_summary = tf.summary.image("tr_image", tr_image)
    tr_fer = tf.placeholder(tf.float32)
    tr_fer_summary = tf.summary.scalar("tr_fer", tr_fer)
    tr_rl = tf.placeholder(tf.float32)
    tr_rl_summary = tf.summary.scalar("tr_rl", tr_rl)
    tr_rw_hist = tf.placeholder(tf.float32)
    tr_rw_hist_summary = tf.summary.histogram("tr_reward_hist", tr_rw_hist)

  with tf.name_scope("per_epoch_eval"):
    best_val_ce = tf.placeholder(tf.float32)
    val_ce = tf.placeholder(tf.float32)
    best_val_ce_summary = tf.summary.scalar("best_valid_ce", best_val_ce)
    val_ce_summary = tf.summary.scalar("valid_ce", val_ce)

  vf = LinearVF()

  if args.parallel:
    if args.aggr_reward:
      gen_episodes = aggr_skip_rnn_act_parallel
    else:
      gen_episodes = skip_rnn_act_parallel
  else:
    gen_episodes = skip_rnn_act

  with tf.Session() as sess:
    sess.run(init_op)

    if args.start_from_ckpt:
      save_op = tf.train.import_meta_graph(os.path.join(args.log_dir,
                                                        'model.ckpt.meta'))
      save_op.restore(sess, os.path.join(args.log_dir, 'model.ckpt'))
      print("Restore from the last checkpoint. "
            "Restarting from %d step." % global_step.eval())

    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph, flush_secs=5.0)

    tr_ce_sum = 0.; tr_ce_count = 0
    tr_acc_sum = 0; tr_acc_count = 0
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

        new_x, new_y, actions, rewards, action_entropies, new_x_mask, new_reward_mask, output_image = \
            gen_episodes(x, x_mask, y, sess, sg, args)

        advantages,_ = compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args)
        #advantages = rewards - np.sum(rewards)/np.sum(new_reward_mask)
        _feed_states = initial_states(n_batch, args.n_hidden)
        
        [_tr_ml_cost,
         _tr_rl_cost,
         _,
         _,
         pred_idx] = sess.run([tg.ml_cost,
                               tg.rl_cost,
                               ml_op,
                               rl_op,
                               tg.pred_idx],
                              feed_dict={tg.seq_x_data: new_x,
                                         tg.seq_x_mask: new_x_mask,
                                         tg.seq_y_data: new_y,
                                         tg.init_state: _feed_states,
                                         tg.seq_action: actions,
                                         tg.seq_advantage: advantages,
                                         tg.seq_action_mask: new_reward_mask})

        tr_ce_sum += _tr_ml_cost.sum()
        tr_ce_count += new_x_mask.sum()

        pred_idx = expand_pred_idx(actions, x_mask, pred_idx, n_batch, args)
        tr_acc_sum += ((pred_idx == y) * x_mask).sum()
        tr_acc_count += x_mask.sum()

        [_tr_ce_summary,
         _tr_fer_summary,
         _tr_rl_summary,
         _tr_image_summary,
         _tr_rw_hist_summary] = sess.run([tr_ce_summary,
                                          tr_fer_summary,
                                          tr_rl_summary,
                                          tr_image_summary,
                                          tr_rw_hist_summary],
                                         feed_dict={tr_ce: _tr_ml_cost.sum() / new_x_mask.sum(),
                                                    tr_fer: ((pred_idx == y) * x_mask).sum() / x_mask.sum(),
                                                    tr_rl: _tr_rl_cost.sum() / new_reward_mask.sum(),
                                                    tr_image: output_image,
                                                    tr_rw_hist: rewards})
        summary_writer.add_summary(_tr_ce_summary, global_step.eval())
        summary_writer.add_summary(_tr_fer_summary, global_step.eval())
        summary_writer.add_summary(_tr_rl_summary, global_step.eval())
        summary_writer.add_summary(_tr_image_summary, global_step.eval())
        summary_writer.add_summary(_tr_rw_hist_summary, global_step.eval())

        tr_rl_costs.append(_tr_rl_cost.sum() / new_reward_mask.sum())
        tr_action_entropies.append(action_entropies.sum() / new_reward_mask.sum())
        tr_rewards.append(rewards.sum()/new_reward_mask.sum())
                  
        if global_step.eval() % args.display_freq == 0:
          avg_tr_ce = tr_ce_sum / tr_ce_count
          avg_tr_fer = 1. - tr_acc_sum / tr_acc_count
          avg_tr_rl_cost = np.asarray(tr_rl_costs).mean()
          avg_tr_action_entropy = np.asarray(tr_action_entropies).mean()
          avg_tr_reward = np.asarray(tr_rewards).mean()

          print("TRAIN: epoch={} iter={} ml_cost(ce/frame)={:.2f} fer={:.2f} rl_cost={:.4f} reward={:.4f} action_entropy={:.2f} time_taken={:.2f}".format(
              _epoch, global_step.eval(), avg_tr_ce, avg_tr_fer, avg_tr_rl_cost, avg_tr_reward, avg_tr_action_entropy, disp_sw.elapsed()))

          tr_ce_sum = 0.; tr_ce_count = 0
          tr_acc_sum = 0.; tr_acc_count = 0
          tr_rl_costs = []
          tr_action_entropies = []
          tr_rewards = []
          
          disp_sw.reset()

      print('--')
      print('End of epoch {}'.format(_epoch+1))
      epoch_sw.print_elapsed()

      print('Testing')

      # Evaluate the model on the validation set
      val_ce_sum = 0.; val_ce_count = 0
      val_acc_sum = 0; val_acc_count = 0
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

        new_x, new_y, actions, rewards, action_entropies, new_x_mask, new_reward_mask, _ = \
            gen_episodes(x, x_mask, y, sess, sg, args)
        advantages, _ = compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args)

        _feed_states = initial_states(n_batch, args.n_hidden)

        _val_ml_cost, _val_rl_cost, pred_idx = sess.run([tg.ml_cost, tg.rl_cost, tg.pred_idx],
                feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask,
                      tg.seq_y_data: new_y, tg.init_state: _feed_states,
                      tg.seq_action: actions, tg.seq_advantage: advantages, 
                      tg.seq_action_mask: new_reward_mask})
        
        val_ce_sum += _val_ml_cost.sum()
        val_ce_count += new_x_mask.sum()

        pred_idx = expand_pred_idx(actions, x_mask, pred_idx, n_batch, args)
        val_acc_sum += ((pred_idx == y) * x_mask).sum()
        val_acc_count += x_mask.sum()

        val_rl_costs.append(_val_rl_cost.sum() / new_reward_mask.sum())
        val_action_entropies.append(action_entropies.sum() / new_reward_mask.sum())
        val_rewards.append(rewards.sum())

      avg_val_ce = val_ce_sum / val_ce_count
      avg_val_fer = 1. - val_acc_sum / val_acc_count
      avg_val_rl_cost = np.asarray(val_rl_costs).mean()
      avg_val_action_entropy = np.asarray(val_action_entropies).mean()
      avg_val_reward = np.asarray(val_rewards).mean()

      print("VALID: epoch={} ml_cost(ce/frame)={:.2f} fer={:.2f} rl_cost={:.4f} reward={:.4f} action_entropy={:.2f} time_taken={:.2f}".format(
          _epoch, avg_val_ce, avg_val_fer, avg_val_rl_cost, avg_val_reward, avg_val_action_entropy, eval_sw.elapsed()))

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




