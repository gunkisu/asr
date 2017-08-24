#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
from itertools import islice

from mixer import gen_zero_state, feed_init_state

from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data, skip_frames_fixed, StopWatch
import mixer

from utils import Accumulator
import utils
import graph_builder
   
if __name__ == '__main__':
    print(' '.join(sys.argv))

    parser = utils.get_argparser()
    parser.add_argument('--n-skip', default=1, type=int, help='Number of frames to skip')
    parser.add_argument('--n-step', default=20, type=int, help='Truncated BPTT steps')

    args = parser.parse_args()
    print(args)
    utils.prepare_dir(args)
    utils.print_host_info()

    tf.get_variable_scope()._reuse = None

    _seed = args.base_seed + args.add_seed
    tf.set_random_seed(_seed)
    np.random.seed(_seed)

    tg = graph_builder.build_graph_subsample_tbptt(args)
    tvars = tf.trainable_variables()
    print([tvar.name for tvar in tvars])
    print("Model size: {:.2f}M".format(utils.get_model_size(tvars)))

    tg_ml_cost = tf.reduce_mean(tg.ml_cost)
    global_step = tf.Variable(0, trainable=False, name="global_step")

    lr = tf.Variable(args.lr, trainable=False, name="lr")

    ml_opt_func = tf.train.AdamOptimizer(learning_rate=lr)
    ml_grads, _ = tf.clip_by_global_norm(tf.gradients(tg_ml_cost, tvars), clip_norm=1.0)
    ml_op = ml_opt_func.apply_gradients(zip(ml_grads, tvars), global_step=global_step)
    
    tf.add_to_collection('n_skip', args.n_skip)
    tf.add_to_collection('n_hidden', args.n_hidden)

    train_set, valid_set, test_set = utils.prepare_dataset(args)

    init_op = tf.global_variables_initializer()
    save_op = tf.train.Saver(max_to_keep=5)
    best_save_op = tf.train.Saver(max_to_keep=5)

    with tf.name_scope("tr_eval"):
        tr_summary = utils.get_summary('ce cr'.split())
    with tf.name_scope("val_eval"):
        val_summary = utils.get_summary('ce cr fer'.split())

    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph, flush_secs=5.0)

        # ce, accuracy, compression ratio
        accu_list = [Accumulator() for i in range(3)]
        ce, ac, cr = accu_list 

        _best_score = np.iinfo(np.int32).max

        epoch_sw, disp_sw, eval_sw = StopWatch(), StopWatch(), StopWatch()

        # For each epoch 
        for _epoch in xrange(args.n_epoch):
            _n_exp = 0

            epoch_sw.reset(); disp_sw.reset()

            print('--')
            print('Epoch {} training'.format(_epoch+1))

            for accu in accu_list: accu.reset()                    

            # For each batch 
            for batch in train_set.get_epoch_iterator():
                orig_x, orig_x_mask, _, _, orig_y, _ = batch

                for sub_batch in skip_frames_fixed([orig_x, orig_x_mask, orig_y], args.n_skip+1):
                    skip_x, skip_x_mask, skip_y = sub_batch
                    n_batch = skip_x.shape[0] # add batch size
                    _n_exp += n_batch

                    prev_state_fw = np.zeros([n_batch, args.n_layer, 2, args.n_hidden])
                    prev_state_bw = np.zeros([n_batch, args.n_layer, 2, args.n_hidden])

                    for win_idx, win in enumerate(utils.win_iter(sub_batch, args.n_step), start=1):
                        x, x_mask, y = win

                        feed_dict={tg.seq_x_data: x, tg.seq_x_mask: x_mask, tg.seq_y_data: y}

                        mixer.feed_prev_state(feed_dict, tg.init_state_fw, prev_state_fw)
                        mixer.feed_prev_state(feed_dict, tg.init_state_bw, prev_state_bw)

                        ml_cost, output_state_fw, output_state_bw, _ = \
                            sess.run([tg.ml_cost, tg.output_state_fw, tg.output_state_bw, ml_op], feed_dict=feed_dict)

                        output_state_fw = np.transpose(np.asarray(output_state_fw), [2,0,1,3])
                        output_state_bw = np.transpose(np.asarray(output_state_bw), [2,0,1,3])

                        mixer.update_prev_state(prev_state_fw, output_state_fw)

                        ce.add(ml_cost.sum(), x_mask.sum())
         
                    orig_count, comp_count = orig_x_mask.sum(), skip_x_mask.sum()
                    cr.add(float(comp_count)/orig_count, 1)

                    summaries = sess.run([s.s for s in tr_summary],
                        feed_dict={tr_summary.ce.ph: ce.last_avg(), tr_summary.cr.ph: cr.avg()})
                    for s in summaries: summary_writer.add_summary(s, global_step.eval())                                 

                if global_step.eval() % args.display_freq == 0:
                    print("TRAIN: epoch={} iter={} ml_cost(ce/frame)={:.3f} compression={:.2f} time_taken={:.2f}".format(
                            _epoch, global_step.eval(), ce.avg(), cr.avg(), disp_sw.elapsed()))
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
                orig_x, orig_x_mask, _, _, orig_y, _ = batch
                 
                for sub_batch in skip_frames_fixed([orig_x, orig_x_mask, orig_y], args.n_skip+1, return_first=True):
                    skip_x, skip_x_mask, skip_y = sub_batch
                    n_batch = skip_x.shape[0] # add batch size
                    _n_exp += n_batch

                    prev_state_fw = np.zeros([n_batch, args.n_layer, 2, args.n_hidden])
                    prev_state_bw = np.zeros([n_batch, args.n_layer, 2, args.n_hidden])

                    pred_idx_list = []

                    for win_idx, win in enumerate(utils.win_iter(sub_batch, args.n_step), start=1):
                        x, x_mask, y = win

                        feed_dict={tg.seq_x_data: x, tg.seq_x_mask: x_mask, tg.seq_y_data: y}

                        mixer.feed_prev_state(feed_dict, tg.init_state_fw, prev_state_fw)
                        mixer.feed_prev_state(feed_dict, tg.init_state_bw, prev_state_bw)

                        ml_cost, pred_idx, output_state_fw, output_state_bw = \
                            sess.run([tg.ml_cost, tg.pred_idx, tg.output_state_fw, tg.output_state_bw], feed_dict=feed_dict)

                        output_state_fw = np.transpose(np.asarray(output_state_fw), [2,0,1,3])

                        mixer.update_prev_state(prev_state_fw, output_state_fw)
                        mixer.update_prev_state(prev_state_bw, output_state_bw)

                        pred_idx_list.append(pred_idx)
                        
                        ce.add(ml_cost.sum(), x_mask.sum())
                    
                    _, n_seq = orig_y.shape
                    pred_idx = np.concatenate(pred_idx_list, axis=-1)
                    pred_idx = pred_idx.reshape([n_batch, -1]).repeat(args.n_skip+1, axis=1)
                    pred_idx = pred_idx[:,:n_seq]

                    orig_count, comp_count = orig_x_mask.sum(), skip_x_mask.sum()

                    cr.add(float(comp_count)/orig_count, 1)
                    ac.add(((pred_idx == orig_y) * orig_x_mask).sum(), orig_count)

            avg_fer = 1-ac.avg()
            print("VALID: epoch={} ml_cost(ce/frame)={:.3f} fer={:.3f} compression={:.2f} time_taken={:.2f}".format(
                    _epoch, ce.avg(), avg_fer, cr.avg(), eval_sw.elapsed()))

            summaries = sess.run([s.s for s in val_summary],
                feed_dict={val_summary.ce.ph: ce.avg(), val_summary.cr.ph: cr.avg(),
                    val_summary.fer.ph: avg_fer})
            for s in summaries: summary_writer.add_summary(s, global_step.eval())                                 

            # Save model
            if avg_fer < _best_score:
                _best_score = avg_fer
                best_ckpt = best_save_op.save(sess, os.path.join(args.logdir, "best_model.ckpt"), global_step=global_step)
                print("Best checkpoint stored in: %s" % best_ckpt)

            ckpt = save_op.save(sess, os.path.join(args.logdir, "model.ckpt"), global_step=global_step)
            print("Checkpoint stored in: %s" % ckpt)

        summary_writer.close()
