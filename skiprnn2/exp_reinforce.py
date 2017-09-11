#!/usr/bin/env python
from __future__ import print_function

import os
import socket
import sys
import argparse

from itertools import islice

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple

import mixer
from mixer import gen_episode_with_seg_reward
from mixer import categorical_ent, expand_output
from mixer import gen_zero_state, feed_init_state
from model import LinearCell

from data.fuel_utils import create_ivector_datastream
import utils
from utils import Accumulator, get_summary

from libs.utils import sync_data, StopWatch
import graph_builder

if __name__ == '__main__':
    print(' '.join(sys.argv))
    
    parser = utils.get_argparser()
    parser.add_argument('--use-prediction', action='store_true', help='Use predictions when computing rewards')
    args = parser.parse_args()
    print(args)
    utils.prepare_dir(args) 
    utils.print_host_info()

    tf.get_variable_scope()._reuse = None

    _seed = args.base_seed + args.add_seed
    tf.set_random_seed(_seed)
    np.random.seed(_seed)

    tg, sg = graph_builder.build_graph_ri(args)
    tvars = tf.trainable_variables()
    print([tvar.name for tvar in tvars])
    print("Model size: {:.2f}M".format(utils.get_model_size(tvars)))
    
    tg_ml_cost = tf.reduce_mean(tg.ml_cost)
    global_step = tf.Variable(0, trainable=False, name="global_step")
    lr = tf.Variable(args.lr, trainable=False, name="lr")
    lr2 = tf.Variable(args.lr2, trainable=False, name="lr2")
    
    ml_opt_func = tf.train.AdamOptimizer(learning_rate=lr)
    rl_opt_func = tf.train.AdamOptimizer(learning_rate=lr2)

    ml_grads, _ = tf.clip_by_global_norm(tf.gradients(tg_ml_cost, tvars), clip_norm=1.0)
    ml_op = ml_opt_func.apply_gradients(zip(ml_grads, tvars), global_step=global_step)

    tg_rl_cost = tf.reduce_mean(tg.rl_cost)
    tg_rl_cost -= args.beta * tg.seq_action_entropy

    rl_grads = tf.gradients(tg_rl_cost, tvars)
    # do not increase global step -- ml op increases it 
    rl_op = rl_opt_func.apply_gradients(zip(rl_grads, tvars))
    
    tf.add_to_collection('n_fast_action', args.n_fast_action)

    train_set, valid_set, test_set = utils.prepare_dataset(args)

    init_op = tf.global_variables_initializer()
    save_op = tf.train.Saver(max_to_keep=args.n_epoch)
    best_save_op = tf.train.Saver(max_to_keep=args.n_epoch)

    with tf.name_scope("tr_eval"):
        tr_summary = get_summary('ce rl cr image'.split())
    with tf.name_scope("val_eval"):
        val_summary = get_summary('ce rl cr fer image'.split())

    vf = mixer.LinearVF()

    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph, flush_secs=5.0)
    
        # ce, accuracy, rl cost, action entropy, reward, compression ratio
        accu_list = [Accumulator() for i in range(6)]
        ce, ac, rl, ae, rw, cr = accu_list 

        _best_score = np.iinfo(np.int32).max

        epoch_sw, disp_sw, eval_sw = StopWatch(), StopWatch(), StopWatch()
        
        # For each epoch 
        for _epoch in xrange(args.n_epoch):
            _n_exp = 0

            epoch_sw.reset(); disp_sw.reset()
            for accu in accu_list: accu.reset()

            print('Epoch {} training'.format(_epoch+1))
            
            # For each batch 
            for batch in train_set.get_epoch_iterator():
                x, x_mask, _, _, y, _ = batch
                n_batch = x.shape[0]
                _n_exp += n_batch

                new_x, new_y, actions_1hot, rewards, action_entropies, new_x_mask, new_reward_mask, output_image, pred_idx = \
                        gen_episode_with_seg_reward(x, x_mask, y, sess, sg, args)
                orig_count, comp_count, rw_count = x_mask.sum(), new_x_mask.sum(), new_reward_mask.sum()
                
                advantages = mixer.compute_advantage2(new_x, new_x_mask, rewards, new_reward_mask, vf, args)

                zero_state = gen_zero_state(n_batch, args.n_hidden)

                feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask, tg.seq_y_data: new_y, 
                    tg.seq_action: actions_1hot, tg.seq_advantage: advantages, 
                    tg.seq_action_mask: new_reward_mask}
        
                feed_init_state(feed_dict, tg.init_state, zero_state)

                _tr_ml_cost, _tr_rl_cost, _, _ = \
                    sess.run([tg.ml_cost, tg.rl_cost, ml_op, rl_op], feed_dict=feed_dict)
                
                ce.add(_tr_ml_cost.sum(), comp_count)
                cr.add(float(comp_count)/orig_count, 1)
                rl.add(_tr_rl_cost.sum(), rw_count)
                rw.add(rewards.sum(), rw_count)
                ae.add(action_entropies.sum(), rw_count)
 
                summaries = sess.run([s.s for s in tr_summary],
                    feed_dict={tr_summary.ce.ph: ce.last_avg(), tr_summary.rl.ph: rl.last_avg(), 
                        tr_summary.image.ph: output_image, tr_summary.cr.ph: cr.avg()})
                for s in summaries: summary_writer.add_summary(s, global_step.eval())

                if global_step.eval() % args.display_freq == 0:
                    print("TRAIN: epoch={} iter={} ml_cost(ce/frame)={:.3f} rl_cost={:.4f} reward={:.4f} action_entropy={:.2f} compression={:.2f} time_taken={:.2f}".format(
                            _epoch, global_step.eval(), ce.avg(), rl.avg(), rw.avg(), ae.avg(), cr.avg(), disp_sw.elapsed()))
                    
                    for accu in accu_list: accu.reset()
                    disp_sw.reset()

            print('--')
            print('End of epoch {}'.format(_epoch+1))
            epoch_sw.print_elapsed()

            print('Testing')

            # Evaluate the model on the validation set
            for accu in accu_list: accu.reset()

            eval_sw.reset()
            for batch in valid_set.get_epoch_iterator():
                x, x_mask, _, _, y, _ = batch
                n_batch = x.shape[0]

                new_x, new_y, actions_1hot, rewards, action_entropies, new_x_mask, new_reward_mask, output_image, pred_idx = \
                        gen_episode_with_seg_reward(x, x_mask, y, sess, sg, args)
                orig_count, comp_count, rw_count = x_mask.sum(), new_x_mask.sum(), new_reward_mask.sum()

                advantages = mixer.compute_advantage2(new_x, new_x_mask, rewards, new_reward_mask, vf, args)
                
                zero_state = gen_zero_state(n_batch, args.n_hidden)
                feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask, tg.seq_y_data: new_y, 
                    tg.seq_action: actions_1hot, tg.seq_advantage: advantages, 
                    tg.seq_action_mask: new_reward_mask}
                feed_init_state(feed_dict, tg.init_state, zero_state)
                
                ml_cost, rl_cost, pred_idx = sess.run([tg.ml_cost, tg.rl_cost, tg.pred_idx], feed_dict=feed_dict)
                pred_idx = expand_output(actions_1hot, x_mask, new_x_mask, pred_idx.reshape([n_batch, -1]), args.n_fast_action)
                
                ce.add(ml_cost.sum(), comp_count)
                ac.add(((pred_idx == y) * x_mask).sum(), orig_count) 
                cr.add(float(comp_count)/orig_count, 1)
                rl.add(rl_cost.sum(), rw_count)
                ae.add(action_entropies.sum(), rw_count)
                rw.add(rewards.sum(), rw_count)

            avg_fer = 1-ac.avg()
            print("VALID: epoch={} ml_cost(ce/frame)={:.3f} rl_cost={:.4f} fer={:.3f} reward={:.4f} action_entropy={:.2f} compression={:.2f} time_taken={:.2f}".format(
                    _epoch, ce.avg(), rl.avg(), avg_fer, rw.avg(), ae.avg(), cr.avg(), eval_sw.elapsed()))

            summaries = sess.run([s.s for s in val_summary],
                feed_dict={val_summary.ce.ph: ce.avg(), val_summary.fer.ph: avg_fer, 
                    val_summary.rl.ph: rl.avg(), val_summary.image.ph: output_image, val_summary.cr.ph: cr.avg()})
            for s in summaries: summary_writer.add_summary(s, global_step.eval())
                    
            # Save model
            if avg_fer < _best_score:
                _best_score = avg_fer
                best_ckpt = best_save_op.save(sess, os.path.join(args.logdir, "best_model.ckpt"), global_step=global_step)
                print("Best checkpoint stored in: %s" % best_ckpt)
        
            ckpt = save_op.save(sess, os.path.join(args.logdir, "model.ckpt"), global_step=global_step)
            print("Checkpoint stored in: %s" % ckpt)

        summary_writer.close()
