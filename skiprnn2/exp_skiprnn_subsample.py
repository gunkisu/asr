#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import nats2bits
from mixer import insert_item2dict
from mixer import lstm_state
from model import LinearCell
from mixer import save_npz2, gen_zero_state, feed_init_state

from collections import namedtuple, OrderedDict

from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data, skip_frames_fixed, StopWatch

from itertools import islice

import utils

TrainGraph = namedtuple('TrainGraph', 'ml_cost seq_x_data seq_x_mask seq_y_data init_state pred_idx seq_label_probs')
TestGraph = namedtuple('TestGraph', 'step_x_data init_state step_last_state step_label_probs')

StopWatchSet = namedtuple('StopWatchSet', 'epoch disp eval per')

def build_graph(args):
    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input), name='seq_x_data')
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name='seq_x_mask')
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state = tuple( lstm_state(args.n_hidden, l) for l in range(args.n_layer))

        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

    with tf.variable_scope('rnn'):
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(num_units=args.n_hidden, forget_bias=0.0)
           
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(args.n_layer)])

    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
    seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

    seq_label_logits = _label_logit(seq_hid_2d, 'label_logit')   

    y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=args.n_class)

    ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
        labels=y_1hot)
    ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

    pred_idx = tf.argmax(seq_label_logits, axis=1)

    seq_label_probs = tf.nn.softmax(seq_label_logits, name='seq_label_probs')

    # testing graph
    step_h_state, step_last_state = cell(step_x_data, init_state, scope='rnn/multi_rnn_cell')
    step_label_logits = _label_logit(step_h_state, 'label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    train_graph = TrainGraph(ml_cost,
                                                     seq_x_data,
                                                     seq_x_mask,
                                                     seq_y_data,
                                                     init_state,
                                                     pred_idx,
                                                     seq_label_probs)

    test_graph = TestGraph(step_x_data, init_state, step_last_state, step_label_probs)

    return train_graph, test_graph
   
if __name__ == '__main__':
    print(' '.join(sys.argv))

    parser = utils.get_argparser()
    parser.add_argument('--n-skip', default=1, type=int, help='Number of frames to skip')
    args = parser.parse_args()
    print(args)
    utils.prepare_dir(args)
    utils.print_host_info()

    tf.get_variable_scope()._reuse = None

    _seed = args.base_seed + args.add_seed
    tf.set_random_seed(_seed)
    np.random.seed(_seed)

    prefix_name = os.path.join(args.logdir, 'model')
    file_name = '%s.npz' % prefix_name

    eval_summary = OrderedDict() # 

    tg, test_graph = build_graph(args)
    tg_ml_cost = tf.reduce_mean(tg.ml_cost)

    global_step = tf.Variable(0, trainable=False, name="global_step")

    tvars = tf.trainable_variables()

    ml_opt_func = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.99)

    ml_grads, _ = tf.clip_by_global_norm(tf.gradients(tg_ml_cost, tvars), clip_norm=1.0)
    ml_op = ml_opt_func.apply_gradients(zip(ml_grads, tvars), global_step=global_step)
    
    tf.add_to_collection('n_skip', args.n_skip)
    tf.add_to_collection('n_hidden', args.n_hidden)

    train_set, valid_set, test_set = utils.prepare_dataset(args)

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
            save_op = tf.train.import_meta_graph(os.path.join(args.logdir, 'model.ckpt.meta'))
            save_op.restore(sess, os.path.join(args.logdir, 'model.ckpt'))
            print("Restore from the last checkpoint. "
                        "Restarting from %d step." % global_step.eval())

        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph, flush_secs=5.0)

        tr_ce_sum = 0.
        tr_ce_count = 0
        tr_acc_sum = 0
        tr_acc_count = 0
        _best_score = np.iinfo(np.int32).max

        sw = StopWatchSet(StopWatch(), StopWatch(), StopWatch(), StopWatch())
        # For each epoch 
        for _epoch in xrange(args.n_epoch):
            _n_exp = 0

            sw.epoch.reset()
            sw.disp.reset()

            print('--')
            print('Epoch {} training'.format(_epoch+1))
            
            # For each batch 
            for batch in train_set.get_epoch_iterator():
                orig_x, orig_x_mask, _, _, orig_y, _ = batch
                 
                for sub_batch in skip_frames_fixed([orig_x, orig_x_mask, orig_y], args.n_skip+1):
                    x, x_mask, y = sub_batch
                    n_batch, _, _ = x.shape
                    _n_exp += n_batch

                    zero_state = gen_zero_state(n_batch, args.n_hidden)

                    feed_dict={tg.seq_x_data: x, tg.seq_x_mask: x_mask, tg.seq_y_data: y}
                    feed_init_state(feed_dict, tg.init_state, zero_state)

                    _tr_ml_cost, _pred_idx, _ = sess.run([tg.ml_cost, tg.pred_idx, ml_op],
                                     feed_dict=feed_dict)

                    tr_ce_sum += _tr_ml_cost.sum()
                    tr_ce_count += x_mask.sum()
                    _tr_ce_summary, = sess.run([tr_ce_summary], feed_dict={tr_ce: _tr_ml_cost.sum() / x_mask.sum()})
                    summary_writer.add_summary(_tr_ce_summary, global_step.eval())

                    _, n_seq = orig_y.shape
                    _pred_idx = _pred_idx.reshape([n_batch, -1]).repeat(args.n_skip+1, axis=1)
                    _pred_idx = _pred_idx[:,:n_seq]

                    tr_acc_sum += ((_pred_idx == orig_y) * orig_x_mask).sum()
                    tr_acc_count += orig_x_mask.sum()
                                 
                if global_step.eval() % args.display_freq == 0:
                    avg_tr_ce = tr_ce_sum / tr_ce_count
                    avg_tr_fer = 1. - float(tr_acc_sum) / tr_acc_count

                    print("TRAIN: epoch={} iter={} ml_cost(ce/frame)={:.2f} fer={:.2f} time_taken={:.2f}".format(
                            _epoch, global_step.eval(), avg_tr_ce, avg_tr_fer, sw.disp.elapsed()))

                    tr_ce_sum = 0.
                    tr_ce_count = 0
                    tr_acc_sum = 0
                    tr_acc_count = 0
                    sw.disp.reset()

            print('--')
            print('End of epoch {}'.format(_epoch+1))
            sw.epoch.print_elapsed()

            print('Testing')

            # Evaluate the model on the validation set
            val_ce_sum = 0.
            val_ce_count = 0
            val_acc_sum = 0
            val_acc_count = 0

            sw.eval.reset()
            for batch in valid_set.get_epoch_iterator():
                orig_x, orig_x_mask, _, _, orig_y, _ = batch
                 
                for sub_batch in skip_frames_fixed([orig_x, orig_x_mask, orig_y], args.n_skip+1, return_first=True):
                    x, x_mask, y = sub_batch
                    n_batch, _, _ = x.shape

                    zero_state = gen_zero_state(n_batch, args.n_hidden)

                    feed_dict={tg.seq_x_data: x, tg.seq_x_mask: x_mask, tg.seq_y_data: y}
                    feed_init_state(feed_dict, tg.init_state, zero_state)

                    _val_ml_cost, _pred_idx = sess.run([tg.ml_cost, tg.pred_idx],
                                    feed_dict=feed_dict)
                    
                    val_ce_sum += _val_ml_cost.sum()
                    val_ce_count += x_mask.sum()

                    _, n_seq = orig_y.shape
                    _pred_idx = _pred_idx.reshape([n_batch, -1]).repeat(args.n_skip+1, axis=1)
                    _pred_idx = _pred_idx[:,:n_seq]

                    val_acc_sum += ((_pred_idx == orig_y) * orig_x_mask).sum()
                    val_acc_count += orig_x_mask.sum()

            avg_val_ce = val_ce_sum / val_ce_count
            avg_val_fer = 1. - float(val_acc_sum) / val_acc_count

            print("VALID: epoch={} ml_cost(ce/frame)={:.2f} fer={:.2f} time_taken={:.2f}".format(
                    _epoch, avg_val_ce, avg_val_fer, sw.eval.elapsed()))

            _val_ce_summary, = sess.run([val_ce_summary], feed_dict={val_ce: avg_val_ce}) 
            summary_writer.add_summary(_val_ce_summary, global_step.eval())
            
            insert_item2dict(eval_summary, 'val_ce', avg_val_ce)
            insert_item2dict(eval_summary, 'time', sw.eval.elapsed())
            save_npz2(file_name, eval_summary)

            # Save model
            if avg_val_ce < _best_score:
                _best_score = avg_val_ce
                best_ckpt = best_save_op.save(sess, os.path.join(args.logdir,
                                                                                                                 "best_model.ckpt"),
                                                                            global_step=global_step)
                print("Best checkpoint stored in: %s" % best_ckpt)
            ckpt = save_op.save(sess, os.path.join(args.logdir, "model.ckpt"),
                                                    global_step=global_step)
            print("Checkpoint stored in: %s" % ckpt)

            _best_val_ce_summary, = sess.run([best_val_ce_summary],
                                                                feed_dict={best_val_ce: _best_score})
            summary_writer.add_summary(_best_val_ce_summary, global_step.eval())
        summary_writer.close()

        print("Optimization Finished.")
