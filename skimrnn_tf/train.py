'''A Long short-term memory (LSTM) implementation using TensorFlow library.
'''
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import nats2bits
from mixer import insert_item2dict
from model import LinearCell, SkimLSTMModule
from model import LSTMModule
from fuel_train import TrainModel

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
    fwd_act_lgp_list = []
    bwd_act_lgp_list = []
    prev_hid_data = x_data
    prev_hid_mask = x_mask
    for l in range(FLAGS.n_layer):
        # Set input data
        prev_input = tf.concat(values=[prev_hid_data, prev_hid_mask],
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
        curr_hid_data, curr_hid_mask, curr_fwd_act_lgp, curr_bwd_act_lgp = outputs

        # Set next input
        prev_hid_data = curr_hid_data
        prev_hid_mask = curr_hid_mask

        # save action log prob
        fwd_act_lgp_list.append(curr_fwd_act_lgp)
        bwd_act_lgp_list.append(curr_bwd_act_lgp)

    # Set output layer
    with tf.variable_scope('output'):
        output_linear = LinearCell(FLAGS.n_class)

    # Get output logit
    output_logit = output_linear(tf.reshape(prev_hid_data, [-1, 2*FLAGS.n_hidden]))

    # Get one-hot label
    y_1hot = tf.one_hot(tf.reshape(y_idx, [-1]), depth=FLAGS.n_class)

    # Define cross entropy
    ml_frame_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_1hot,
                                                            logits=output_logit)

    ml_sample_loss = tf.reduce_sum(ml_frame_loss * tf.reshape(x_mask, [-1]), axis=0)

    ml_sum_loss = tf.reduce_sum(ml_frame_loss * tf.reshape(x_mask, [-1]))
    ml_mean_loss = ml_sum_loss/tf.reduce_sum(x_mask)


    total_lgp_list = []
    for fwd_lgp, bwd_lgp in zip(fwd_act_lgp_list, bwd_act_lgp_list):
        fwd_sum = tf.reduce_sum(fwd_lgp, axis=[0, 2])
        bwd_sum = tf.reduce_sum(bwd_lgp, axis=[0, 2])
        fwd_cnt = tf.reduce_sum(tf.to_float(tf.not_equal(fwd_lgp, 0.0)), axis=[0, 2])
        bwd_cnt = tf.reduce_sum(tf.to_float(tf.not_equal(bwd_lgp, 0.0)), axis=[0, 2])




    return (x_data,
            x_mask,
            y_idx,
            init_state,
            init_cntr,
            sum_loss,
            mean_loss,
            fwd_log_act_list,
            bwd_log_act_list)


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
