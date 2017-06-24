#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '..')
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple
from mixer import insert_item2dict
from mixer import save_npz2
from mixer import improve_skip_rnn_act_parallel
from mixer import LinearVF, discount
from mixer import categorical_ent
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
flags.DEFINE_integer('min-read', 3, 'Number of min read after skip done')
# flags.DEFINE_integer('n-fast-action', 10, 'Number of steps to skip in the fast action mode')
flags.DEFINE_integer('base-seed', 20170309, 'Base random seed')
flags.DEFINE_integer('add-seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start-from-ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad-clip', True, 'If true, clip the gradients')
# flags.DEFINE_boolean('parallel', True, 'If true, do parallel sampling')
# flags.DEFINE_boolean('fast-action', False, 'If true, operate in the fast action mode')
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
flags.DEFINE_boolean('use-final-reward', False, '')
flags.DEFINE_float('ml-l2', 0.0, 'ml l2 lambda')
flags.DEFINE_boolean('use-baseline', True, '')

tg_fields = ['seq_x_data',
             'seq_x_mask',
             'seq_y_data',
             'seq_action_data',
             'seq_action_mask',
             'seq_advantage',
             'seq_reward',
             'seq_label_logits',
             'seq_ml_cost',
             'seq_rl_cost',
             'seq_real_rl_cost',
             'seq_action_ent']

sg_fields = ['step_x_data',
             'prev_states',
             'step_h_state',
             'step_last_state',
             'step_label_probs',
             'step_action_probs',
             'step_action_samples']

TrainGraph = namedtuple('TrainGraph', ' '.join(tg_fields))
SampleGraph = namedtuple('SampleGraph', ' '.join(sg_fields))


# Build graph
def build_graph(args):
    ##################
    # Input variable #
    ##################
    with tf.device(args.device):
        ##################
        # Sequence-level #
        ##################
        # Input sequence data [batch_size, seq_len, ...]
        seq_x_data = tf.placeholder(dtype=tf.float32,
                                    shape=(None, None, args.n_input),
                                    name='seq_x_data')
        seq_x_mask = tf.placeholder(dtype=tf.float32,
                                    shape=(None, None),
                                    name='seq_x_mask')
        seq_y_data = tf.placeholder(dtype=tf.int32,
                                    shape=(None, None),
                                    name='seq_y_data')

        # Action related data [batch_size, seq_len, ...]
        seq_action_data = tf.placeholder(dtype=tf.float32,
                                         shape=(None, None, args.n_action),
                                         name='seq_action_data')
        seq_action_mask = tf.placeholder(dtype=tf.float32,
                                         shape=(None, None),
                                         name='seq_action_mask')
        seq_advantage = tf.placeholder(dtype=tf.float32,
                                       shape=(None, None),
                                       name='seq_advantage')
        seq_reward = tf.placeholder(dtype=tf.float32,
                                    shape=(None, None),
                                    name='seq_reward')
        ##############
        # Step-level #
        ##############
        # Input step data [batch_size, n_input]
        step_x_data = tf.placeholder(dtype=tf.float32,
                                     shape=(None, args.n_input),
                                     name='step_x_data')

        # Prev state [2, batch_size, n_hidden]
        prev_state = tf.placeholder(dtype=tf.float32,
                                    shape=(2, None, args.n_hidden),
                                    name='prev_states')
    ###########
    # Modules #
    ###########
    # Recurrent Module (LSTM)
    with tf.variable_scope('rnn'):
        _rnn = LSTMModule(num_units=args.n_hidden)

    # Labelling Module (FF)
    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class, activation=None)

    # Actioning Module (FF)
    with tf.variable_scope('action'):
        _action_logit = LinearCell(num_units=args.n_action, activation=None)

    ##################
    # Sampling graph #
    ##################
    # Recurrent update
    step_h_state, step_last_state = _rnn(inputs=step_x_data,
                                         init_state=prev_state,
                                         one_step=True)

    # Label logits/probs
    step_label_logits = _label_logit(inputs=step_h_state,
                                     scope='label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits)

    # Action logits
    if FLAGS.ref_input:
        step_action_input = [step_x_data, step_h_state]
    else:
        step_action_input = step_h_state
    step_action_logits = _action_logit(inputs=step_action_input,
                                       scope='action_logit')

    # Action probs
    step_action_probs = tf.nn.softmax(logits=step_action_logits)

    # Action sampling
    step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1)

    # Set sampling graph
    sample_graph = SampleGraph(step_x_data,
                               prev_state,
                               step_h_state,
                               step_last_state,
                               step_label_probs,
                               step_action_probs,
                               step_action_samples)

    ##################
    # Training graph #
    ##################
    # Recurrent update
    init_state = tf.zeros(shape=(2, tf.shape(seq_x_data)[0], args.n_hidden))
    seq_h_state_3d, seq_last_state = _rnn(inputs=seq_x_data,
                                          init_state=init_state,
                                          one_step=False)

    # Label logits/probs
    seq_label_logits = _label_logit(inputs=tf.reshape(seq_h_state_3d, [-1, args.n_hidden]),
                                    scope='label_logit')

    # Action logits
    if FLAGS.ref_input:
        seq_action_input = [tf.reshape(seq_x_data, [-1, args.n_input]),
                            tf.reshape(seq_h_state_3d, [-1, args.n_hidden])]
    else:
        seq_action_input = tf.reshape(seq_h_state_3d, [-1, args.n_hidden])
    seq_action_logits = _action_logit(inputs=seq_action_input,
                                      scope='action_logit')
    # Action probs
    seq_action_probs = tf.nn.softmax(logits=seq_action_logits)

    # Action entropy
    seq_action_ent = categorical_ent(dist=seq_action_probs)*tf.reshape(seq_action_mask, [-1])

    # ML cost (logP(label))
    seq_y_1hot = tf.one_hot(indices=tf.reshape(seq_y_data, [-1]),
                            depth=args.n_class)
    seq_ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits,
                                                          labels=seq_y_1hot)
    seq_ml_cost *= tf.reshape(seq_x_mask, [-1])

    # RL cost (logP(action)*reward)
    seq_rl_cost = -tf.log(seq_action_probs+1e-8) * tf.reshape(seq_action_data, [-1, args.n_action])
    seq_rl_cost = tf.reduce_sum(seq_rl_cost, axis=-1)
    seq_rl_cost *= tf.reshape(seq_advantage, [-1]) * tf.reshape(seq_action_mask, [-1])

    # RL cost wo/ baseline
    seq_real_rl_cost = tf.reshape(seq_reward, [-1])*tf.reshape(seq_action_mask, [-1])

    # Set training graph
    train_graph = TrainGraph(seq_x_data,
                             seq_x_mask,
                             seq_y_data,
                             seq_action_data,
                             seq_action_mask,
                             seq_advantage,
                             seq_reward,
                             seq_label_logits,
                             seq_ml_cost,
                             seq_rl_cost,
                             seq_real_rl_cost,
                             seq_action_ent)

    return train_graph, sample_graph


def expand_pred_idx(seq_skip_1hot,
                    seq_skip_mask,
                    seq_prd_idx,
                    seq_x_mask):
    # Init output
    expand_output = np.zeros_like(seq_x_mask)

    # Get Step size
    seq_skip_step = np.argmax(seq_skip_1hot, axis=2) + 1

    # For each data
    for i, (skip_step, skip_mask, prd_idx) in enumerate(zip(seq_skip_step, seq_skip_mask, seq_prd_idx)):

        # For each step
        start_idx = 0
        for j, (s, m, p) in enumerate(zip(skip_step, skip_mask, prd_idx)):
            if m == 1.:
                end_idx = start_idx + s
            else:
                end_idx = start_idx + 1

            expand_output[i, start_idx:end_idx] = p
            start_idx = end_idx

    return expand_output

def compute_advantage(seq_h_data,
                      seq_r_data,
                      seq_r_mask,
                      vf,
                      args,
                      final_cost=False):
    discounted_rewards = []

    # For each sample data
    for reward, mask in zip(seq_r_data, seq_r_mask):
        # Collect rewards
        reward_val = []
        reward_pos = []
        for i, (r, m) in enumerate(zip(reward, mask)):
            if m:
                reward_val.append(r)
                reward_pos.append(i)

        # Compute discounted reward
        discounted_reward_val = discount(np.array(reward_val), args.discount_gamma)

        # If using only final reward
        if final_cost:
            discounted_reward_val = np.ones_like(discounted_reward_val)*discounted_reward_val[0]

        # Put reward value to right position
        reward_val = np.ones_like(reward)
        for i, r in zip(reward_pos, discounted_reward_val):
            reward_val[i] = r
        discounted_rewards.append(reward_val)
    discounted_rewards = np.array(discounted_rewards)
    # Compute baseline
    seq_h_data_1d = seq_h_data.reshape([-1, seq_h_data.shape[2]])
    baseline_1d = vf.predict(seq_h_data_1d)
    baseline_2d = baseline_1d.reshape([seq_h_data.shape[0], -1]) * seq_r_mask

    # Compute modified reward based on baseline
    advantages = discounted_rewards - baseline_2d
    mean_advantages = np.mean(advantages*seq_r_mask)
    std_advantages = np.std(advantages*seq_r_mask)
    advantages = ((advantages - mean_advantages) / (std_advantages+1e-8)) * seq_r_mask

    valid_idx = np.where(seq_r_mask==1.)
    valid_h = seq_h_data[valid_idx]
    valid_r = discounted_rewards[valid_idx]

    vf.fit(valid_h, valid_r)

    return advantages, discounted_rewards

# Main function
def main(_):
    # Print settings
    print(' '.join(sys.argv))
    args = FLAGS
    print(args.__flags)

    # Load checkpoint
    if not args.start_from_ckpt:
        if tf.gfile.Exists(args.log_dir):
            tf.gfile.DeleteRecursively(args.log_dir)
        tf.gfile.MakeDirs(args.log_dir)

    # ???
    tf.get_variable_scope()._reuse = None

    # Set random seed
    _seed = args.base_seed + args.add_seed
    tf.set_random_seed(_seed)
    np.random.seed(_seed)

    # Set save file name
    prefix_name = os.path.join(args.log_dir, 'model')
    file_name = '%s.npz' % prefix_name

    # Set evaluation summary
    eval_summary = OrderedDict()

    # Build model graph
    tg, sg = build_graph(args)

    # Set linear regressor for baseline
    vf = LinearVF()

    # Set global step
    global_step = tf.Variable(0, trainable=False, name="global_step")

    # Get ml/rl related parameters
    tvars = tf.trainable_variables()
    ml_vars = [tvar for tvar in tvars if "action" not in tvar.name]
    rl_vars = [tvar for tvar in tvars if "action" in tvar.name]

    # Set optimizer
    ml_opt_func = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    rl_opt_func = tf.train.AdamOptimizer(learning_rate=args.rl_learning_rate)

    # Set model ml cost (sum over all and divide it by batch_size)
    ml_cost = tf.reduce_sum(tg.seq_ml_cost)
    ml_cost /= tf.to_float(tf.shape(tg.seq_x_data)[0])
    ml_cost += args.ml_l2*0.5*tf.add_n([tf.reduce_sum(tf.square(var)) for var in ml_vars])

    # Set model rl cost (sum over all and divide it by batch_size, also entropy cost)
    rl_cost = tg.seq_rl_cost - args.ent_weight*tg.seq_action_ent
    rl_cost = tf.reduce_sum(rl_cost)
    rl_cost /= tf.to_float(tf.shape(tg.seq_x_data)[0])

    # Set model rl cost (sum over all and divide it by batch_size, also entropy cost)
    real_rl_cost = tf.reduce_sum(tg.seq_real_rl_cost)
    real_rl_cost /= tf.reduce_sum(tg.seq_action_mask)

    # Gradient clipping for ML
    ml_grads = tf.gradients(ml_cost, ml_vars)
    if args.grad_clip:
        ml_grads, _ = tf.clip_by_global_norm(t_list=ml_grads,
                                             clip_norm=1.0)

    # Gradient for RL
    rl_grads = tf.gradients(rl_cost, rl_vars)

    # ML optimization
    ml_op = ml_opt_func.apply_gradients(grads_and_vars=zip(ml_grads, ml_vars),
                                        global_step=global_step,
                                        name='ml_op')

    # RL optimization
    rl_op = rl_opt_func.apply_gradients(grads_and_vars=zip(rl_grads, rl_vars),
                                        global_step=global_step,
                                        name='rl_op')

    # Sync dataset
    sync_data(args)

    # Get dataset
    train_set = create_ivector_datastream(path=args.data_path,
                                          which_set=args.train_dataset,
                                          batch_size=args.batch_size,
                                          min_after_cache=args.min_after_cache,
                                          length_sort=not args.no_length_sort)
    valid_set = create_ivector_datastream(path=args.data_path,
                                          which_set=args.valid_dataset,
                                          batch_size=args.batch_size,
                                          min_after_cache=args.min_after_cache,
                                          length_sort=not args.no_length_sort)

    # Set param init op
    init_op = tf.global_variables_initializer()

    # Set save op
    save_op = tf.train.Saver(max_to_keep=5)
    best_save_op = tf.train.Saver(max_to_keep=5)

    # Set per-step logging
    with tf.name_scope("per_step_eval"):
        # For ML cost (ce)
        tr_ce = tf.placeholder(tf.float32)
        tr_ce_summary = tf.summary.scalar("train_ml_cost", tr_ce)

        # For output visualization
        tr_image = tf.placeholder(tf.float32)
        tr_image_summary = tf.summary.image("train_image", tr_image)

        # For ML FER
        tr_fer = tf.placeholder(tf.float32)
        tr_fer_summary = tf.summary.scalar("train_fer", tr_fer)

        # For RL cost
        tr_rl = tf.placeholder(tf.float32)
        tr_rl_summary = tf.summary.scalar("train_rl", tr_rl)

        # For RL reward
        tr_reward = tf.placeholder(tf.float32)
        tr_reward_summary = tf.summary.scalar("train_reward", tr_reward)

        # For RL entropy
        tr_ent = tf.placeholder(tf.float32)
        tr_ent_summary = tf.summary.scalar("train_entropy", tr_ent)

        # For RL reward histogram
        tr_rw_hist = tf.placeholder(tf.float32)
        tr_rw_hist_summary = tf.summary.histogram("train_reward_hist", tr_rw_hist)

    # Set per-epoch logging
    with tf.name_scope("per_epoch_eval"):
        # For best valid ML cost (full)
        best_val_ce = tf.placeholder(tf.float32)
        best_val_ce_summary = tf.summary.scalar("best_valid_ce", best_val_ce)

        # For best valid FER
        best_val_fer = tf.placeholder(tf.float32)
        best_val_fer_summary = tf.summary.scalar("best_valid_fer", best_val_fer)

        # For valid ML cost (full)
        val_ce = tf.placeholder(tf.float32)
        val_ce_summary = tf.summary.scalar("valid_ce", val_ce)

        # For valid FER
        val_fer = tf.placeholder(tf.float32)
        val_fer_summary = tf.summary.scalar("valid_fer", val_fer)

    # Set module
    gen_episodes = improve_skip_rnn_act_parallel

    # Init session
    with tf.Session() as sess:
        # Init model
        sess.run(init_op)

        # Load from checkpoint
        if args.start_from_ckpt:
          save_op = tf.train.import_meta_graph(os.path.join(args.log_dir, 'model.ckpt.meta'))
          save_op.restore(sess, os.path.join(args.log_dir, 'model.ckpt'))
          print("Restore from the last checkpoint. Restarting from %d step." % global_step.eval())

        # Summary writer
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph, flush_secs=5.0)

        # For train tracking
        tr_ce_sum = 0.; tr_ce_count = 0
        tr_acc_sum = 0.; tr_acc_count = 0
        tr_rl_sum = 0.; tr_rl_count = 0
        tr_ent_sum = 0.; tr_ent_count = 0
        tr_reward_sum = 0.; tr_reward_count = 0

        _best_ce = np.iinfo(np.int32).max
        _best_fer = 1.00

        # For time measure
        epoch_sw = StopWatch()
        disp_sw = StopWatch()
        eval_sw = StopWatch()

        # For each epoch
        for _epoch in xrange(args.n_epoch):
            # Reset timer
            epoch_sw.reset()
            disp_sw.reset()
            print('Epoch {} training'.format(_epoch + 1))

            # Set rl skipping flag
            use_rl_skipping = True #if _best_fer < 0.5 else False

            # For each batch (update)
            for batch_data in train_set.get_epoch_iterator():
                ##################
                # Sampling Phase #
                ##################
                # Get batch data
                seq_x_data, seq_x_mask, _, _, seq_y_data, _ = batch_data

                # Use skipping
                if use_rl_skipping:
                    # Transpose axis
                    seq_x_data = np.transpose(seq_x_data, (1, 0, 2))
                    seq_x_mask = np.transpose(seq_x_mask, (1, 0))
                    seq_y_data = np.transpose(seq_y_data, (1, 0))

                    # Number of samples
                    batch_size = seq_x_data.shape[1]
                    # Sample actions (episode generation)
                    [skip_x_data,
                     skip_h_data,
                     skip_x_mask,
                     skip_y_data,
                     skip_action_data,
                     skip_action_mask,
                     skip_rewards,
                     result_image] = gen_episodes(seq_x_data=seq_x_data,
                                                  seq_x_mask=seq_x_mask,
                                                  seq_y_data=seq_y_data,
                                                  sess=sess,
                                                  sample_graph=sg,
                                                  args=args)

                    # Compute baseline and refine reward
                    skip_advantage, skip_disc_rewards = compute_advantage(seq_h_data=skip_h_data,
                                                                          seq_r_data=skip_rewards,
                                                                          seq_r_mask=skip_action_mask,
                                                                          vf=vf,
                                                                          args=args,
                                                                          final_cost=args.use_final_reward)

                    if args.use_baseline is False:
                        skip_advantage = skip_disc_rewards

                    ##################
                    # Training Phase #
                    ##################
                    # Update model
                    [_tr_ml_cost,
                     _tr_rl_cost,
                     _,
                     _,
                     _tr_act_ent,
                     _tr_pred_logit] = sess.run([ml_cost,
                                                 real_rl_cost,
                                                 ml_op,
                                                 rl_op,
                                                 tg.seq_action_ent,
                                                 tg.seq_label_logits],
                                                feed_dict={tg.seq_x_data: skip_x_data,
                                                           tg.seq_x_mask: skip_x_mask,
                                                           tg.seq_y_data: skip_y_data,
                                                           tg.seq_action_data: skip_action_data,
                                                           tg.seq_action_mask: skip_action_mask,
                                                           tg.seq_advantage: skip_advantage,
                                                           tg.seq_reward: skip_disc_rewards})

                    seq_x_mask = np.transpose(seq_x_mask, (1, 0))
                    seq_y_data = np.transpose(seq_y_data, (1, 0))

                    # Get full sequence prediction
                    _tr_pred_full = expand_pred_idx(seq_skip_1hot=skip_action_data,
                                                    seq_skip_mask=skip_action_mask,
                                                    seq_prd_idx=_tr_pred_logit.argmax(axis=-1).reshape([batch_size, -1]),
                                                    seq_x_mask=seq_y_data)

                    # Update history
                    tr_ce_sum += _tr_ml_cost.sum() * batch_size
                    tr_ce_count += skip_x_mask.sum()

                    tr_acc_sum += ((_tr_pred_full == seq_y_data) * seq_x_mask).sum()
                    tr_acc_count += seq_x_mask.sum()

                    tr_rl_sum += _tr_rl_cost.sum()
                    tr_rl_count += 1.0

                    tr_ent_sum += _tr_act_ent.sum()
                    tr_ent_count += skip_action_mask.sum()

                    tr_reward_sum += (skip_rewards*skip_action_mask).sum()
                    tr_reward_count += skip_action_mask.sum()

                    ################
                    # Write result #
                    ################
                    [_tr_rl_summary,
                     _tr_image_summary,
                     _tr_ent_summary,
                     _tr_reward_summary,
                     _tr_rw_hist_summary] = sess.run([tr_rl_summary,
                                                      tr_image_summary,
                                                      tr_ent_summary,
                                                      tr_reward_summary,
                                                      tr_rw_hist_summary],
                                                     feed_dict={tr_rl: _tr_rl_cost.sum(),
                                                                tr_image: result_image,
                                                                tr_ent: (_tr_act_ent.sum() / skip_action_mask.sum()),
                                                                tr_reward: ((skip_rewards*skip_action_mask).sum()/skip_action_mask.sum()),
                                                                tr_rw_hist: skip_rewards})
                    summary_writer.add_summary(_tr_rl_summary, global_step.eval())
                    summary_writer.add_summary(_tr_image_summary, global_step.eval())
                    summary_writer.add_summary(_tr_ent_summary, global_step.eval())
                    summary_writer.add_summary(_tr_reward_summary, global_step.eval())
                    summary_writer.add_summary(_tr_rw_hist_summary, global_step.eval())
                else:
                    # Number of samples
                    batch_size = seq_x_data.shape[0]

                    ##################
                    # Training Phase #
                    ##################
                    [_tr_ml_cost,
                     _,
                     _tr_pred_full] = sess.run([ml_cost,
                                                ml_op,
                                                tg.seq_label_logits],
                                               feed_dict={tg.seq_x_data: seq_x_data,
                                                          tg.seq_x_mask: seq_x_mask,
                                                          tg.seq_y_data: seq_y_data})
                    _tr_pred_full = np.reshape(_tr_pred_full.argmax(axis=1), seq_y_data.shape)

                    # Update history
                    tr_ce_sum += _tr_ml_cost.sum()*batch_size
                    tr_ce_count += seq_x_mask.sum()

                    tr_acc_sum += ((_tr_pred_full == seq_y_data) * seq_x_mask).sum()
                    tr_acc_count += seq_x_mask.sum()

                    skip_x_mask = seq_x_mask

                ################
                # Write result #
                ################
                [_tr_ce_summary,
                 _tr_fer_summary] = sess.run([tr_ce_summary,
                                              tr_fer_summary],
                                             feed_dict={tr_ce: (_tr_ml_cost.sum()*batch_size) / skip_x_mask.sum(),
                                                        tr_fer: ((_tr_pred_full == seq_y_data) * seq_x_mask).sum() / seq_x_mask.sum()})
                summary_writer.add_summary(_tr_ce_summary, global_step.eval())
                summary_writer.add_summary(_tr_fer_summary, global_step.eval())

                # Display results
                if global_step.eval() % args.display_freq == 0:
                    # Get average results
                    avg_tr_ce = tr_ce_sum / tr_ce_count
                    avg_tr_fer = 1. - tr_acc_sum / tr_acc_count
                    if use_rl_skipping:
                        avg_tr_rl = tr_rl_sum / tr_rl_count
                        avg_tr_ent = tr_ent_sum / tr_ent_count
                        avg_tr_reward = tr_reward_sum / tr_reward_count
                        print("TRAIN: epoch={} iter={} "
                              "ml_cost(ce/frame)={:.2f} fer={:.2f} "
                              "rl_cost={:.4f} reward={:.4f} action_entropy={:.2f} "
                              "time_taken={:.2f}".format(_epoch, global_step.eval(),
                                                         avg_tr_ce, avg_tr_fer,
                                                         avg_tr_rl, avg_tr_reward, avg_tr_ent,
                                                         disp_sw.elapsed()))
                    else:
                        print("TRAIN: epoch={} iter={} "
                              "ml_cost(ce/frame)={:.2f} fer={:.2f} "
                              "time_taken={:.2f}".format(_epoch, global_step.eval(),
                                                         avg_tr_ce, avg_tr_fer,
                                                         disp_sw.elapsed()))

                    # Reset average results
                    tr_ce_sum = 0.; tr_ce_count = 0
                    tr_acc_sum = 0.; tr_acc_count = 0
                    tr_rl_sum = 0.; tr_rl_count = 0
                    tr_ent_sum = 0.; tr_ent_count = 0
                    tr_reward_sum = 0.; tr_reward_count = 0

                    disp_sw.reset()

            # End of epoch
            print('--')
            print('End of epoch {}'.format(_epoch+1))
            epoch_sw.print_elapsed()

            # Evaluation
            print('Testing')

            # Evaluate the model on the validation set
            val_ce_sum = 0.; val_ce_count = 0
            val_acc_sum = 0.; val_acc_count = 0
            val_rl_sum = 0.; val_rl_count = 0
            val_ent_sum = 0.; val_ent_count = 0
            val_reward_sum = 0.; val_reward_count = 0
            eval_sw.reset()

            # For each batch in Valid
            for batch_data in valid_set.get_epoch_iterator():
                ##################
                # Sampling Phase #
                ##################
                # Get batch data
                seq_x_data, seq_x_mask, _, _, seq_y_data, _ = batch_data

                if use_rl_skipping:
                    # Transpose axis
                    seq_x_data = np.transpose(seq_x_data, (1, 0, 2))
                    seq_x_mask = np.transpose(seq_x_mask, (1, 0))
                    seq_y_data = np.transpose(seq_y_data, (1, 0))

                    # Number of samples
                    batch_size = seq_x_data.shape[1]

                    # Sample actions (episode generation)
                    [skip_x_data,
                     skip_h_data,
                     skip_x_mask,
                     skip_y_data,
                     skip_action_data,
                     skip_action_mask,
                     skip_rewards,
                     result_image] = gen_episodes(seq_x_data=seq_x_data,
                                                  seq_x_mask=seq_x_mask,
                                                  seq_y_data=seq_y_data,
                                                  sess=sess,
                                                  sample_graph=sg,
                                                  args=args)

                    # Compute baseline and refine reward
                    skip_advantage, skip_disc_rewards = compute_advantage(seq_h_data=skip_h_data,
                                                                          seq_r_data=skip_rewards,
                                                                          seq_r_mask=skip_action_mask,
                                                                          vf=vf,
                                                                          args=args,
                                                                          final_cost=args.use_final_reward)

                    if args.use_baseline is False:
                        skip_advantage = skip_disc_rewards

                    #################
                    # Forward Phase #
                    #################
                    # Update model
                    [_val_ml_cost,
                     _val_rl_cost,
                     _val_pred_logit,
                     _val_action_ent] = sess.run([ml_cost,
                                                  real_rl_cost,
                                                  tg.seq_label_logits,
                                                  tg.seq_action_ent],
                                                 feed_dict={tg.seq_x_data: skip_x_data,
                                                            tg.seq_x_mask: skip_x_mask,
                                                            tg.seq_y_data: skip_y_data,
                                                            tg.seq_action_data: skip_action_data,
                                                            tg.seq_action_mask: skip_action_mask,
                                                            tg.seq_advantage: skip_advantage,
                                                            tg.seq_reward: skip_disc_rewards})

                    seq_x_mask = np.transpose(seq_x_mask, (1, 0))
                    seq_y_data = np.transpose(seq_y_data, (1, 0))

                    # Get full sequence prediction
                    _val_pred_full = expand_pred_idx(seq_skip_1hot=skip_action_data,
                                                     seq_skip_mask=skip_action_mask,
                                                     seq_prd_idx=_val_pred_logit.argmax(axis=-1).reshape([batch_size, -1]),
                                                     seq_x_mask=seq_y_data)

                    # Update history
                    val_ce_sum += _val_ml_cost.sum() * batch_size
                    val_ce_count += skip_x_mask.sum()

                    val_acc_sum += ((_val_pred_full == seq_y_data) * seq_x_mask).sum()
                    val_acc_count += seq_x_mask.sum()

                    val_rl_sum += _val_rl_cost.sum()
                    val_rl_count += 1.0

                    val_ent_sum += _val_action_ent.sum()
                    val_ent_count += skip_action_mask.sum()

                    val_reward_sum += (skip_rewards*skip_action_mask).sum()
                    val_reward_count += skip_action_mask.sum()
                else:
                    # Number of samples
                    batch_size = seq_x_data.shape[0]

                    #################
                    # Forward Phase #
                    #################
                    # Update model
                    [_val_ml_cost,
                     _val_pred_full] = sess.run([ml_cost,
                                                 tg.seq_label_logits],
                                                 feed_dict={tg.seq_x_data: seq_x_data,
                                                            tg.seq_x_mask: seq_x_mask,
                                                            tg.seq_y_data: seq_y_data})
                    _val_pred_full = np.reshape(_val_pred_full.argmax(axis=1), seq_y_data.shape)

                    # Update history
                    val_ce_sum += _val_ml_cost.sum()*batch_size
                    val_ce_count += seq_x_mask.sum()

                    val_acc_sum += ((_val_pred_full == seq_y_data) * seq_x_mask).sum()
                    val_acc_count += seq_x_mask.sum()

            # Aggregate over all valid data
            avg_val_ce = val_ce_sum / val_ce_count
            avg_val_fer = 1. - val_acc_sum / val_acc_count

            if use_rl_skipping:
                avg_val_rl = val_rl_sum / val_rl_count
                avg_val_ent = val_ent_sum / val_ent_count
                avg_val_reward = val_reward_sum / val_reward_count

                print("VALID: epoch={} "
                      "ml_cost(ce/frame)={:.2f} fer={:.2f} "
                      "rl_cost={:.4f} reward={:.4f} action_entropy={:.2f} "
                      "time_taken={:.2f}".format(_epoch,
                                                 avg_val_ce, avg_val_fer,
                                                 avg_val_rl, avg_val_reward, avg_val_ent,
                                                 eval_sw.elapsed()))
            else:
                print("VALID: epoch={} "
                      "ml_cost(ce/frame)={:.2f} fer={:.2f} "
                      "time_taken={:.2f}".format(_epoch,
                                                 avg_val_ce, avg_val_fer,
                                                 eval_sw.elapsed()))

            [_val_ce_summary,
             _val_fer_summary] = sess.run([val_ce_summary,
                                           val_fer_summary],
                                          feed_dict={val_ce: avg_val_ce,
                                                     val_fer: avg_val_fer})
            summary_writer.add_summary(_val_ce_summary, global_step.eval())
            summary_writer.add_summary(_val_fer_summary, global_step.eval())

            insert_item2dict(eval_summary, 'val_ce', avg_val_ce)
            insert_item2dict(eval_summary, 'val_fer', avg_val_fer)
            # insert_item2dict(eval_summary, 'val_rl', avg_val_rl)
            # insert_item2dict(eval_summary, 'val_reward', avg_val_reward)
            # insert_item2dict(eval_summary, 'val_ent', avg_val_ent)
            insert_item2dict(eval_summary, 'time', eval_sw.elapsed())
            save_npz2(file_name, eval_summary)

            # Save best model
            if avg_val_ce < _best_ce:
                _best_ce = avg_val_ce
                best_ckpt = best_save_op.save(sess=sess,
                                              save_path=os.path.join(args.log_dir, "best_model(ce).ckpt"),
                                              global_step=global_step)
                print("Best checkpoint based on CE stored in: %s" % best_ckpt)

            if avg_val_fer < _best_fer:
                _best_fer = avg_val_fer
                best_ckpt = best_save_op.save(sess=sess,
                                              save_path=os.path.join(args.log_dir, "best_model(fer).ckpt"),
                                              global_step=global_step)
                print("Best checkpoint based on FER stored in: %s" % best_ckpt)

            # Save model
            ckpt = save_op.save(sess=sess,
                                save_path=os.path.join(args.log_dir, "model.ckpt"),
                                global_step=global_step)
            print("Checkpoint stored in: %s" % ckpt)

            # Write result
            [_best_val_ce_summary,
             _best_val_fer_summary] = sess.run([best_val_ce_summary,
                                                best_val_fer_summary],
                                               feed_dict={best_val_ce: _best_ce,
                                                          best_val_fer: _best_fer})
            summary_writer.add_summary(_best_val_ce_summary, global_step.eval())
            summary_writer.add_summary(_best_val_fer_summary, global_step.eval())

        # Done of training
        summary_writer.close()
        print("Optimization Finished.")

if __name__ == '__main__':
  tf.app.run()




