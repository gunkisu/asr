'''A Long short-term memory (LSTM) implementation using TensorFlow library.
'''
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import nats2bits
from mixer import insert_item2dict
from model import LinearCell, SkimLSTMModule
from collections import namedtuple

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay factor')
flags.DEFINE_integer('batch_size', 64, 'Size of mini-batch')
flags.DEFINE_integer('n_epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('display_freq', 100, 'Display frequency')
flags.DEFINE_integer('max_seq_len', 100, 'Maximum length of sequences')
flags.DEFINE_integer('n_input', 123, 'Number of RNN hidden units')
flags.DEFINE_integer('n_hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n_class', 3436, 'Number of target symbols')
flags.DEFINE_integer('base_seed', 20170309, 'Base random seed')
flags.DEFINE_integer('add_seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start_from_ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad_clip', True, 'If true, clip the gradients')
flags.DEFINE_boolean('eval_train', False, 'If true, evaluate on train set')
flags.DEFINE_boolean('use_layer_norm', False, 'If true, apply layer norm')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('log_dir', 'wsj_lstm', 'Directory path to files')
flags.DEFINE_boolean('no_copy', False, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data_path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('train_dataset', 'train_si284', '')
flags.DEFINE_string('valid_dataset', 'test_dev93', '')
flags.DEFINE_string('test_dataset', 'test_eval92', '')

Graph = namedtuple('Graph', 'cost x x_mask state last_state y')

def build_graph(FLAGS):
    """Define training graph.
    """

    # Define input data
    with tf.device(FLAGS.device):
        x_data = tf.placeholder(dtype=tf.float32,
                                shape=(None, None, FLAGS.n_input),
                                name='x_data')
        x_mask = tf.placeholder(dtype=tf.float32,
                                shape=(None, None, 1),
                                name='x_mask')
        y_idx = tf.placeholder(dtype=tf.int32,
                               shape=(None, None),
                               name='y_idx')
        init_state = tf.placeholder(dtype=tf.float32,
                                    shape=(None, FLAGS.n_hidden),
                                    name='init_state')
        init_cntr = tf.placeholder(dtype=tf.float32,
                                   shape=(None, 1),
                                   name='init_cntr')

    # Define model
    fwd_hid_list = []
    fwd_act_lgp_list = []
    fwd_act_mask_list = []

    bwd_hid_list = []
    bwd_act_lgp_list = []
    bwd_act_mask_list = []
    prev_hid_data = x_data
    for l in range(FLAGS.n_layer):
        # Set input data
        prev_input = tf.concat(values=[prev_hid_data, x_mask],
                               axis=-1,
                               name='input_{}'.format(l))

        # Set skim lstm
        with tf.variable_scope('lstm_{}'.format(l)):
            skim_lstm = SkimLSTMModule(num_units=FLAGS.n_hidden,
                                       max_skims=FLAGS.n_action,
                                       min_reads=FLAGS.n_read,
                                       forget_bias=FLAGS.forget_bias)

        # Run bidir skim lstm
        outputs = skim_lstm(inputs=prev_input,
                            init_state=[init_state, init_cntr],
                            use_bidir=True)

        # Get output
        cur_hid_data, cur_read_mask, cur_act_mask, curr_act_lgp = outputs

        # Set next input
        prev_hid_data = cur_hid_data

        # save hidden
        cur_fwd_hid, cur_bwd_hid = tf.split(cur_hid_data, num_or_size_splits=2, axis=-1)
        fwd_hid_list.append(cur_fwd_hid)
        bwd_hid_list.append(cur_bwd_hid)

        # save action mask
        cur_fwd_act_mask, cur_bwd_act_mask = tf.split(cur_act_mask, num_or_size_splits=2, axis=-1)
        fwd_act_mask_list.append(cur_fwd_act_mask)
        bwd_act_mask_list.append(cur_bwd_act_mask)

        # save action log prob
        cur_fwd_act_lgp, cur_bwd_act_lgp = tf.split(curr_act_lgp, num_or_size_splits=2, axis=-1)
        fwd_act_lgp_list.append(cur_fwd_act_lgp)
        bwd_act_lgp_list.append(cur_bwd_act_lgp)

    # Set output layer
    with tf.variable_scope('output'):
        output_linear = LinearCell(FLAGS.n_class)

    # Get sequence length and batch size
    seq_len = tf.shape(prev_hid_data)[0]
    num_samples = tf.shape(prev_hid_data)[1]
    output_feat_size = tf.shape(prev_hid_data)[2]

    # Get output logit
    output_logit = output_linear(tf.reshape(prev_hid_data, [-1, output_feat_size]))
    output_logit = tf.reshape(output_logit, (seq_len, num_samples, FLAGS.n_class))

    # Get one-hot label
    y_1hot = tf.one_hot(y_idx, depth=FLAGS.n_class)

    # Define cross entropy
    ml_frame_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y_1hot, [-1, FLAGS.n_class]),
                                                            logits=tf.reshape(output_logit, [-1, FLAGS.n_class]))

    ml_sample_loss = tf.reshape(ml_frame_loss, (seq_len, num_samples))
    ml_sample_loss = tf.reduce_sum(ml_sample_loss*tf.squeeze(x_mask, axis=-1), axis=0)/tf.reduce_sum(x_mask, axis=[0, 1])
    ml_mean_loss = tf.reduce_sum(ml_sample_loss)/num_samples

    # Define frame-wise accuracy
    sample_frame_accr = tf.to_float(tf.equal(tf.argmax(output_logit, axis=-1), tf.argmax(y_1hot, axis=-1)))
    sample_frame_accr = tf.reduce_sum(sample_frame_accr*tf.squeeze(x_mask, axis=-1))/tf.reduce_sum(x_mask, axis=[0, 1])
    mean_frame_accr = tf.reduce_sum(sample_frame_accr)/num_samples

    # Get parameters
    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    rl_params = [var for var in model_params if 'action' in var.name]
    ml_params = [var for var in model_params if 'action' not in var.name]

    # Define RL cost
    sample_reward = sample_frame_accr
    total_policy_cost = []
    total_baseline_cost = []
    for i, act_data_list in enumerate(zip(fwd_hid_list,
                                          fwd_act_lgp_list,
                                          fwd_act_mask_list,
                                          bwd_hid_list,
                                          bwd_act_lgp_list,
                                          bwd_act_mask_list)):

        fwd_hid, fwd_lgp, fwd_mask, bwd_hid, bwd_lgp, bwd_mask = act_data_list
        # Forward pass
        # Get action mask and corresponding hidden state
        with tf.variable_scope('fwd_baseline') as vs:
            fwd_W = tf.get_variable('W', [FLAGS.n_hidden, 1], dtype=fwd_hid.dtype)
            fwd_b = tf.get_variable('b', [FLAGS.n_hidden, 1], dtype=fwd_hid.dtype)
            tf.add_to_collection('weights', fwd_W)
            tf.add_to_collection('vars', fwd_W)
            tf.add_to_collection('vars', fwd_b)

        # set baseline
        fwd_basline = tf.matmul(tf.reshape(fwd_hid, [-1, FLAGS.n_hidden]), fwd_W) + fwd_b
        fwd_basline = tf.reshape(fwd_basline, [seq_len, num_samples])

        # set sample-wise reward
        fwd_sample_reward = (sample_reward - fwd_basline)*tf.squeeze(fwd_mask)

        # set baseline cost
        rl_fwd_baseline_cost = tf.reduce_sum(tf.square(fwd_sample_reward))
        total_baseline_cost.append([rl_fwd_baseline_cost, [fwd_W, fwd_b]])

        # set policy cost
        rl_fwd_policy_cost = fwd_sample_reward*tf.reduce_sum(fwd_lgp, axis=-1)*tf.squeeze(fwd_mask)
        rl_fwd_policy_cost = tf.reduce_sum(rl_fwd_policy_cost)/tf.reduce_sum(fwd_mask)
        total_policy_cost.append([rl_fwd_policy_cost, [var for var in rl_params if str(i) in var and 'fwd' in var]])

        # Backward pass
        # Get action mask and corresponding hidden state
        with tf.variable_scope('bwd_baseline') as vs:
            bwd_W = tf.get_variable('W', [FLAGS.n_hidden, 1], dtype=bwd_hid.dtype)
            bwd_b = tf.get_variable('b', [FLAGS.n_hidden, 1], dtype=bwd_hid.dtype)
            tf.add_to_collection('weights', bwd_W)
            tf.add_to_collection('vars', bwd_W)
            tf.add_to_collection('vars', bwd_b)

        # set baseline
        bwd_basline = tf.matmul(tf.reshape(bwd_hid, [-1, FLAGS.n_hidden]), bwd_W) + bwd_b
        bwd_basline = tf.reshape(bwd_basline, [seq_len, num_samples])

        # set sample-wise reward
        bwd_sample_reward = (sample_reward - bwd_basline)*tf.squeeze(bwd_mask)

        # set baseline cost
        rl_bwd_baseline_cost = tf.reduce_sum(tf.square(bwd_sample_reward))
        total_baseline_cost.append([rl_bwd_baseline_cost, [bwd_W, bwd_b]])

        # set policy cost
        rl_bwd_policy_cost = bwd_sample_reward*tf.reduce_sum(bwd_lgp, axis=-1)*tf.squeeze(bwd_mask)
        rl_bwd_policy_cost = tf.reduce_sum(rl_bwd_policy_cost)/tf.reduce_sum(bwd_mask)
        total_policy_cost.append([rl_bwd_policy_cost, [var for var in rl_params if str(i) in var and 'bwd' in var]])

    ml_cost = [ml_mean_loss, ml_params]

    return (x_data,
            x_mask,
            y_idx,
            init_state,
            init_cntr,
            ml_mean_loss,
            mean_frame_accr,
            ml_cost,
            total_policy_cost,
            total_baseline_cost)


def initial_states(batch_size, n_hidden):
  init_state = np.zeros([2, batch_size, n_hidden], dtype=np.float32)
  return init_state


def sess_wrapper(x, x_mask, y, t_step, step_tr_cost, sess, G, recursive_states):
  _tr_cost, _, _state, _step_tr_cost = \
      sess.run([G[0], t_step, G[4], step_tr_cost],
               feed_dict={G[1]: x, G[2]: x_mask,
                          G[3]: recursive_states,
						  G.y: y})
  return _tr_cost, _step_tr_cost, _state


def monitor(G, sess, train_set, valid_set, FLAGS,
            summary, S, summary_writer, _epoch, monitor_ops=None):
  # Process training set
  if train_set is None:
    tr_nats = 0
    tr_bits = 0
  else:
    _step = 0
    _cost = 0
    _len = 0
    for batch in train_set:
      x, x_mask, _, _, y, _ = batch
      x = np.transpose(x, (1, 0, 2))
      x_mask = np.transpose(x_mask, (1, 0))
      y = np.transpose(y, (1, 0))

      _, n_batch, _ = x.shape

      _state = initial_states(n_batch, FLAGS.n_hidden)

      _tr_cost, _state = sess.run([G[0], G[4]],
                    feed_dict={G[1]: x, G[2]: x_mask, G[3]: _state, G.y: y})
      _cost += _tr_cost.sum()
      _len += x_mask.sum()
    tr_nats = _cost / _len
    tr_bits = nats2bits(tr_nats)
  # Process validation set
  _step = 0
  _cost = 0
  _len = 0
  for batch in valid_set:
    x, x_mask, _, _, y, _ = batch
    x = np.transpose(x, (1, 0, 2))
    x_mask = np.transpose(x_mask, (1, 0))
    y = np.transpose(y, (1, 0))

    _, n_batch, _ = x.shape

    _state = initial_states(n_batch, FLAGS.n_hidden)
    _val_cost, _state = sess.run([G[0], G[4]],
                  feed_dict={G[1]: x, G[2]: x_mask, G[3]: _state, G.y: y})
    _cost += _val_cost.sum()
    _len += x_mask.sum()
  val_nats = _cost / _len
  val_bits = nats2bits(val_nats)
  # Write summary logs
  _epoch_tr_bits, _epoch_val_bits,  = sess.run([S[0][1], S[0][2]],
                                               feed_dict={S[1][1]: tr_bits,
                                                          S[1][2]: val_bits})
  if train_set is not None:
    summary_writer.add_summary(_epoch_tr_bits, _epoch)
  summary_writer.add_summary(_epoch_val_bits, _epoch)
  insert_item2dict(summary, 'tr_nats', tr_nats)
  insert_item2dict(summary, 'tr_bits', tr_bits)
  insert_item2dict(summary, 'val_nats', val_nats)
  insert_item2dict(summary, 'val_bits', val_bits)
  return tr_nats, tr_bits, val_nats, val_bits


def define_summary():
  with tf.name_scope("per_epoch_eval"):
    # Add epoch level monitoring channels to summary
    best_val_bits = tf.placeholder(tf.float32, [])
    tr_bits = tf.placeholder(tf.float32, [])
    val_bits = tf.placeholder(tf.float32, [])
    best_epoch_val_bits = tf.summary.scalar("best_valid_bits", best_val_bits)
    epoch_tr_bits = tf.summary.scalar("train_bits", tr_bits)
    epoch_val_bits = tf.summary.scalar("valid_bits", val_bits)
  return ([best_epoch_val_bits, epoch_tr_bits, epoch_val_bits],
          [best_val_bits, tr_bits, val_bits])


def main(_):
  if not FLAGS.start_from_ckpt:
    if tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
  train = TrainModel(FLAGS, build_graph, monitor, initial_states, sess_wrapper,
                     define_summary)
  train()


if __name__ == '__main__':
  tf.app.run()
  main(_)
