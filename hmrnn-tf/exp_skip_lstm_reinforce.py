#!/usr/bin/env python

import os
import socket
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
from mixer import get_gpuname
from mixer import gen_episode_with_seg_reward
from mixer import LinearVF, compute_advantage
from mixer import categorical_ent, expand_output
from mixer import lstm_state, gen_zero_state, feed_init_state
from model import LinearCell

from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data, StopWatch

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning-rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('rl-learning-rate', 0.01, 'Initial learning rate for RL')
flags.DEFINE_float('beta', 1.0, 'beta in reward computation')
flags.DEFINE_float('alpha', 1.0, 'alpha in reward computation')
flags.DEFINE_integer('min-after-cache', 1024, 'Size of mini-batch')
flags.DEFINE_integer('n-batch', 64, 'Size of mini-batch')
flags.DEFINE_integer('n-epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('display-freq', 100, 'Display frequency')
flags.DEFINE_integer('n-input', 123, 'Number of RNN hidden units')
flags.DEFINE_integer('n-layer', 1, 'Number of RNN hidden layers')
flags.DEFINE_integer('n-hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n-class', 3436, 'Number of target symbols')
flags.DEFINE_integer('n-action', 3, 'Number of actions (max skim size)')
flags.DEFINE_integer('n-fast-action', 10, 'Number of steps to skip in the fast action mode')
flags.DEFINE_integer('base-seed', 20170309, 'Base random seed') 
flags.DEFINE_integer('add-seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start-from-ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad-clip', True, 'If true, clip the gradients')
flags.DEFINE_boolean('fast-action', False, 'If true, operate in the fast action mode')
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

tg_fields = ['ml_cost', 'rl_cost', 'seq_x_data', 'seq_x_mask',
    'seq_y_data', 'init_state', 'seq_action', 'seq_advantage', 'seq_action_mask', 'pred_idx']

sg_fields = ['step_h_state', 'step_last_state', 'step_label_probs', 'step_action_probs',
    'step_action_samples', 'step_x_data', 'init_state', 'action_entropy']

TrainGraph = namedtuple('TrainGraph', ' '.join(tg_fields))
SampleGraph = namedtuple('SampleGraph', ' '.join(sg_fields))

def build_graph(args):
    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input))
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state = tuple( lstm_state(args.n_hidden, l) for l in range(args.n_layer))

        seq_action = tf.placeholder(tf.float32, shape=(None, None, args.n_action))
        seq_advantage = tf.placeholder(tf.float32, shape=(None, None))
        seq_action_mask = tf.placeholder(tf.float32, shape=(None, None))
        
        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

    def lstm_cell():
        return tf.contrib.rnn.LSTMCell(num_units=args.n_hidden, forget_bias=0.0)
       
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(args.n_layer)])

    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class)

    with tf.variable_scope('action'):
        _action_logit = LinearCell(num_units=args.n_action)

    # sampling graph
    step_h_state, step_last_state = cell(step_x_data, init_state, scope='rnn/multi_rnn_cell')

    step_label_logits = _label_logit(step_h_state, 'label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    step_action_logits = _action_logit(step_h_state, 'action_logit')
    step_action_probs = tf.nn.softmax(logits=step_action_logits, name='step_action_probs')
    step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1, name='step_action_samples')
    step_action_entropy = categorical_ent(step_action_probs)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
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

    seq_action_logits = _action_logit(seq_hid_2d_rl, 'action_logit')
    seq_action_probs = tf.nn.softmax(seq_action_logits)

    action_prob_entropy = categorical_ent(seq_action_probs)
    action_prob_entropy *= tf.reshape(seq_action_mask, [-1])
    action_prob_entropy = tf.reduce_sum(action_prob_entropy)/tf.reduce_sum(seq_action_mask)

    rl_cost = tf.reduce_sum(tf.log(seq_action_probs+1e-8) \
        * tf.reshape(seq_action, [-1,args.n_action]), axis=-1)
    rl_cost *= tf.reshape(seq_advantage, [-1])
    rl_cost = -tf.reduce_sum(rl_cost*tf.reshape(seq_action_mask, [-1]))

    train_graph = TrainGraph(ml_cost, rl_cost, seq_x_data, seq_x_mask, 
        seq_y_data, init_state, seq_action, seq_advantage, seq_action_mask, pred_idx)

    sample_graph = SampleGraph(step_h_state, step_last_state, step_label_probs,
        step_action_probs, step_action_samples, step_x_data, init_state, step_action_entropy)

    return train_graph, sample_graph

def main(_):
    print(' '.join(sys.argv))
    args = FLAGS
    print(args.__flags)
    print('Hostname: {}'.format(socket.gethostname()))
    print('GPU: {}'.format(get_gpuname()))

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
    print([tvar.name for tvar in tvars])
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
    # do not increase global step -- ml op increases it 
    rl_op = rl_opt_func.apply_gradients(zip(rl_grads, rl_vars))
    
    tf.add_to_collection('fast_action', args.fast_action)
    tf.add_to_collection('fast_action', args.n_fast_action)

    sync_data(args)
    datasets = [args.train_dataset, args.valid_dataset, args.test_dataset]
    train_set, valid_set, test_set = [create_ivector_datastream(path=args.data_path, which_set=dataset, 
            batch_size=args.n_batch, min_after_cache=args.min_after_cache, length_sort=not args.no_length_sort) for dataset in datasets]

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

    with tf.Session() as sess:
        sess.run(init_op)

        if args.start_from_ckpt:
            save_op = tf.train.import_meta_graph(os.path.join(args.log_dir, 'model.ckpt.meta'))
            save_op.restore(sess, os.path.join(args.log_dir, 'model.ckpt'))
            print("Restore from the last checkpoint. Restarting from %d step." % global_step.eval())

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
                n_batch = x.shape[0]
                _n_exp += n_batch

                new_x, new_y, actions_1hot, rewards, action_entropies, new_x_mask, new_reward_mask, output_image = \
                        gen_episode_with_seg_reward(x, x_mask, y, sess, sg, args)
                advantages,_ = compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args)

                zero_state = gen_zero_state(n_batch, args.n_hidden)

                feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask, tg.seq_y_data: new_y, 
                    tg.seq_action: actions_1hot, tg.seq_advantage: advantages, 
                    tg.seq_action_mask: new_reward_mask}
                feed_init_state(feed_dict, tg.init_state, zero_state)

                _tr_ml_cost, _tr_rl_cost, _, _, pred_idx = \
                    sess.run([tg.ml_cost, tg.rl_cost, ml_op, rl_op, tg.pred_idx], feed_dict=feed_dict)
        
                tr_ce_sum += _tr_ml_cost.sum()
                tr_ce_count += new_x_mask.sum()

                pred_idx = expand_output(actions_1hot, x_mask, new_x_mask, pred_idx.reshape([n_batch, -1]))
                tr_acc_sum += ((pred_idx == y) * x_mask).sum()
                tr_acc_count += x_mask.sum()

                _tr_ce_summary, _tr_fer_summary, _tr_rl_summary, _tr_image_summary, _tr_rw_hist_summary = \
                    sess.run([tr_ce_summary, tr_fer_summary, tr_rl_summary, tr_image_summary, tr_rw_hist_summary],
                        feed_dict={tr_ce: _tr_ml_cost.sum() / new_x_mask.sum(), 
                            tr_fer: ((pred_idx == y) * x_mask).sum() / x_mask.sum(), 
                            tr_rl: _tr_rl_cost.sum() / new_reward_mask.sum(),
                            tr_image: output_image, tr_rw_hist: rewards})
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
                n_batch = x.shape[0]

                new_x, new_y, actions_1hot, rewards, action_entropies, new_x_mask, new_reward_mask, _ = \
                        gen_episode_with_seg_reward(x, x_mask, y, sess, sg, args)
                advantages, _ = compute_advantage(new_x, new_x_mask, rewards, new_reward_mask, vf, args)

                zero_state = gen_zero_state(n_batch, args.n_hidden)

                feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask, tg.seq_y_data: new_y, 
                    tg.seq_action: actions_1hot, tg.seq_advantage: advantages, 
                    tg.seq_action_mask: new_reward_mask}
                feed_init_state(feed_dict, tg.init_state, zero_state)

                _val_ml_cost, _val_rl_cost, pred_idx = sess.run([tg.ml_cost, tg.rl_cost, tg.pred_idx],
                                feed_dict=feed_dict)
                
                val_ce_sum += _val_ml_cost.sum()
                val_ce_count += new_x_mask.sum()

                pred_idx = expand_output(actions_1hot, x_mask, new_x_mask, pred_idx.reshape([n_batch, -1]))
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




