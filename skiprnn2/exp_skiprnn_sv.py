#!/usr/bin/env python
from __future__ import print_function

import os
import socket
import sys

from itertools import islice

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple
from mixer import gen_mask
from mixer import insert_item2dict
from mixer import save_npz2
from mixer import get_gpuname
from mixer import gen_supervision, skip_rnn_forward_supervised, gen_episode_supervised
from mixer import LinearVF, compute_advantage
from mixer import categorical_ent, expand_output
from mixer import lstm_state, gen_zero_state, feed_init_state
from model import LinearCell

from data.fuel_utils import create_ivector_datastream
import utils

from libs.utils import sync_data, StopWatch

TrainGraph = namedtuple('TrainGraph', 
    'ml_cost rl_cost seq_x_data seq_x_mask seq_y_data seq_jump_data init_state pred_idx')
TestGraph = namedtuple('TestGraph', 
    'step_h_state step_last_state step_label_probs step_action_probs step_pred_idx step_x_data init_state')

def build_graph(args):
    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input))
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state = tuple( lstm_state(args.n_hidden, l) for l in range(args.n_layer))

        seq_jump_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

    def lstm_cell():
        return tf.contrib.rnn.LSTMCell(num_units=args.n_hidden, forget_bias=0.0)
       
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(args.n_layer)])

    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class)

    with tf.variable_scope('action'):
        _action_logit = LinearCell(num_units=args.n_action)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
    seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

    seq_label_logits = _label_logit(seq_hid_2d, 'label_logit')

    y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=args.n_class)

    ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
        labels=y_1hot)
    ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

    pred_idx = tf.argmax(seq_label_logits, axis=1)

    seq_hid_3d_rl = seq_hid_3d[:,:-1,:]
    seq_hid_2d_rl = tf.reshape(seq_hid_3d_rl, [-1, args.n_hidden])
    seq_hid_2d_rl = tf.stop_gradient(seq_hid_2d_rl)

    seq_action_logits = _action_logit(seq_hid_2d_rl, 'action_logit')
    seq_action_probs = tf.nn.softmax(seq_action_logits)

    jump_1hot = tf.one_hot(tf.reshape(seq_jump_data, [-1]), depth=args.n_action)

    rl_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_action_probs, 
        labels=jump_1hot)
    rl_cost = tf.reduce_sum(rl_cost*tf.reshape(seq_x_mask[:,:-1], [-1]))

    train_graph = TrainGraph(ml_cost, rl_cost, seq_x_data, seq_x_mask, 
        seq_y_data, seq_jump_data, init_state, pred_idx)

    # testing graph (step_h_state == step_last_state)
    step_h_state, step_last_state = cell(step_x_data, init_state, scope='rnn/multi_rnn_cell')

    step_label_logits = _label_logit(step_h_state, 'label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    step_action_logits = _action_logit(step_h_state, 'action_logit')
    step_action_probs = tf.nn.softmax(logits=step_action_logits, name='step_action_probs')

    step_pred_idx = tf.argmax(step_action_logits, axis=1, name='step_pred_idx')
    
    test_graph = TestGraph(step_h_state, step_last_state, step_label_probs, 
        step_action_probs, step_pred_idx, step_x_data, init_state)

    return train_graph, test_graph

if __name__ == '__main__':
    print(' '.join(sys.argv))

    args = utils.get_argparser().parse_args()
    print(args)
    utils.prepare_dir(args)
    utils.print_host_info()

    tf.get_variable_scope()._reuse = None

    _seed = args.base_seed + args.add_seed
    tf.set_random_seed(_seed)
    np.random.seed(_seed)

    prefix_name = os.path.join(args.logdir, 'model')
    file_name = '%s.npz' % prefix_name

    eval_summary = OrderedDict()

    tg, test_graph = build_graph(args)
    tg_ml_cost = tf.reduce_mean(tg.ml_cost)

    global_step = tf.Variable(0, trainable=False, name="global_step")

    tvars = tf.trainable_variables()
    print([tvar.name for tvar in tvars])

    ml_opt_func = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.99)
    rl_opt_func = tf.train.AdamOptimizer(learning_rate=args.rl_learning_rate, beta1=0.9, beta2=0.99)

    ml_grads, _ = tf.clip_by_global_norm(tf.gradients(tg_ml_cost, tvars), clip_norm=1.0)
    ml_op = ml_opt_func.apply_gradients(zip(ml_grads, tvars), global_step=global_step)

    tg_rl_cost = tf.reduce_mean(tg.rl_cost)
    rl_grads = tf.gradients(tg_rl_cost, tvars)
    rl_op = rl_opt_func.apply_gradients(zip(rl_grads, tvars))
    
    tf.add_to_collection('n_fast_action', args.n_fast_action)
    
    train_set, valid_set, test_set = utils.prepare_dataset(args)

    init_op = tf.global_variables_initializer()
    save_op = tf.train.Saver(max_to_keep=5)
    best_save_op = tf.train.Saver(max_to_keep=5)

    with tf.name_scope("per_step_eval"):
        tr_ce = tf.placeholder(tf.float32)
        tr_ce_summary = tf.summary.scalar("tr_ce", tr_ce)
        tr_fer = tf.placeholder(tf.float32)
        tr_fer_summary = tf.summary.scalar("tr_fer", tr_fer)
        tr_ce2 = tf.placeholder(tf.float32)
        tr_ce2_summary = tf.summary.scalar("tr_rl", tr_ce2)

        tr_image = tf.placeholder(tf.float32)
        tr_image_summary = tf.summary.image("tr_image", tr_image)

    with tf.name_scope("per_epoch_eval"):
        val_fer = tf.placeholder(tf.float32)
        val_fer_summary = tf.summary.scalar("val_fer", val_fer)
        best_val_fer = tf.placeholder(tf.float32)
        best_val_fer_summary = tf.summary.scalar("best_valid_fer", best_val_fer)
        val_image = tf.placeholder(tf.float32)
        val_image_summary = tf.summary.image("val_image", val_image)

    vf = LinearVF()

    with tf.Session() as sess:
        sess.run(init_op)

        if args.start_from_ckpt:
            save_op = tf.train.import_meta_graph(os.path.join(args.logdir, 'model.ckpt.meta'))
            save_op.restore(sess, os.path.join(args.logdir, 'model.ckpt'))
            print("Restore from the last checkpoint. Restarting from %d step." % global_step.eval())

        summary_writer = tf.summary.FileWriter(args.logdir, sess.graph, flush_secs=5.0)

        tr_ce_sum = 0.; tr_ce_count = 0
        tr_acc_sum = 0; tr_acc_count = 0
        tr_ce2_sum = 0.; tr_ce2_count = 0

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


                new_x, new_y, actions, actions_1hot, new_x_mask = gen_supervision(x, x_mask, y, args)
                
                zero_state = gen_zero_state(n_batch, args.n_hidden)

                feed_dict={tg.seq_x_data: new_x, tg.seq_x_mask: new_x_mask, tg.seq_y_data: new_y, 
                    tg.seq_jump_data: actions}
                feed_init_state(feed_dict, tg.init_state, zero_state)

                _tr_ml_cost, _tr_rl_cost, _, _ = \
                    sess.run([tg.ml_cost, tg.rl_cost, ml_op, rl_op], feed_dict=feed_dict)
        
                tr_ce_sum += _tr_ml_cost.sum()
                tr_ce_count += new_x_mask.sum()
                tr_ce2_sum += _tr_rl_cost.sum()
                tr_ce2_count += new_x_mask[:,:-1].sum()

                actions_1hot, label_probs, new_mask, output_image = \
                    skip_rnn_forward_supervised(x, x_mask, sess, test_graph, args.n_fast_action, y)
                
                pred_idx = expand_output(actions_1hot, x_mask, new_mask, label_probs.argmax(axis=-1))
                tr_acc_sum += ((pred_idx == y) * x_mask).sum()
                tr_acc_count += x_mask.sum()

                _tr_ce_summary, _tr_fer_summary, _tr_ce2_summary, _tr_image_summary = \
                    sess.run([tr_ce_summary, tr_fer_summary, tr_ce2_summary, tr_image_summary],
                        feed_dict={tr_ce: _tr_ml_cost.sum() / new_x_mask.sum(), 
                            tr_fer: 1 - ((pred_idx == y) * x_mask).sum() / x_mask.sum(), 
                            tr_ce2: _tr_rl_cost.sum() / new_x_mask[:,:-1].sum(),
                            tr_image: output_image})
                summary_writer.add_summary(_tr_ce_summary, global_step.eval())
                summary_writer.add_summary(_tr_fer_summary, global_step.eval())
                summary_writer.add_summary(_tr_ce2_summary, global_step.eval())
                summary_writer.add_summary(_tr_image_summary, global_step.eval())
            

                if global_step.eval() % args.display_freq == 0:
                    avg_tr_ce = tr_ce_sum / tr_ce_count
                    avg_tr_fer = 1. - tr_acc_sum / tr_acc_count
                    avg_tr_ce2 = tr_ce2_sum / tr_ce2_count

                    print("TRAIN: epoch={} iter={} ml_cost(ce/frame)={:.2f} fer={:.2f} rl_cost={:.4f} time_taken={:.2f}".format(
                            _epoch, global_step.eval(), avg_tr_ce, avg_tr_fer, avg_tr_ce2, disp_sw.elapsed()))

                    tr_ce_sum = 0.; tr_ce_count = 0
                    tr_acc_sum = 0.; tr_acc_count = 0
                    tr_ce2_sum = 0.; tr_ce2_count = 0
                    
                    disp_sw.reset()

            print('--')
            print('End of epoch {}'.format(_epoch+1))
            epoch_sw.print_elapsed()

            print('Testing')

            # Evaluate the model on the validation set
            val_acc_sum = 0; val_acc_count = 0
            
            eval_sw.reset()
            for batch in valid_set.get_epoch_iterator():
                x, x_mask, _, _, y, _ = batch
                n_batch = x.shape[0]

                actions_1hot, label_probs, new_mask, output_image = \
                    skip_rnn_forward_supervised(x, x_mask, sess, test_graph, args.n_fast_action, y)
                
                pred_idx = expand_output(actions_1hot, x_mask, new_mask, label_probs.argmax(axis=-1))
                val_acc_sum += ((pred_idx == y) * x_mask).sum()
                val_acc_count += x_mask.sum()

            avg_val_fer = 1. - val_acc_sum / val_acc_count
         
            print("VALID: epoch={} fer={:.2f} time_taken={:.2f}".format(
                    _epoch, avg_val_fer, eval_sw.elapsed()))

            _val_fer_summary, _val_image_summary = sess.run([val_fer_summary, val_image_summary], 
                feed_dict={val_fer: avg_val_fer, val_image: output_image}) 
            summary_writer.add_summary(_val_fer_summary, global_step.eval())
            summary_writer.add_summary(_val_image_summary, global_step.eval())
           
            insert_item2dict(eval_summary, 'val_fer', avg_val_fer)
            insert_item2dict(eval_summary, 'time', eval_sw.elapsed())
            save_npz2(file_name, eval_summary)

            # Save model
            if avg_val_fer < _best_score:
                _best_score = avg_val_fer
                best_ckpt = best_save_op.save(sess, os.path.join(args.logdir,
                                                                                                                 "best_model.ckpt"),
                                                                            global_step=global_step)
                print("Best checkpoint stored in: %s" % best_ckpt)
            ckpt = save_op.save(sess, os.path.join(args.logdir, "model.ckpt"),
                                                    global_step=global_step)
            print("Checkpoint stored in: %s" % ckpt)

            _best_val_fer_summary, = sess.run([best_val_fer_summary],
                                                                feed_dict={best_val_fer: _best_score})
            summary_writer.add_summary(_best_val_fer_summary, global_step.eval())
        summary_writer.close()

        print("Optimization Finished.")



