'''A Long short-term memory (LSTM) implementation using TensorFlow library.
'''
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import nats2bits
from model import LinearCell
from model import LSTMModule
from ptb_train import TrainModel


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay factor')
flags.DEFINE_integer('batch_size', 64, 'Size of mini-batch')
flags.DEFINE_integer('n_epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('display_freq', 100, 'Display frequency')
flags.DEFINE_integer('max_seq_len', 100, 'Maximum length of sequences')
flags.DEFINE_integer('n_hidden', 1024, 'Number of RNN hidden units')
flags.DEFINE_integer('n_class', 50, 'Number of target symbols')
flags.DEFINE_integer('n_input_embed', 128, 'Number of input embedding dimension')
flags.DEFINE_integer('base_seed', 20170309, 'Base random seed')
flags.DEFINE_integer('add_seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start_from_ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad_clip', True, 'If true, clip the gradients')
flags.DEFINE_boolean('eval_train', False, 'If true, evaluate on train set')
flags.DEFINE_boolean('use_layer_norm', False, 'If true, apply layer norm')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('log_dir', '/raid/chungjun/nips2017/testbed/ptb_lstm', 'Directory path to files')


def build_graph(FLAGS):
  """Define training graph.
  """
  with tf.device(FLAGS.device):
    # Graph input
    inp = tf.placeholder(tf.int32, [FLAGS.max_seq_len, None])
    inp_mask = tf.placeholder(tf.float32, [FLAGS.max_seq_len, None])
    state = tf.placeholder(tf.float32, [2, None, FLAGS.n_hidden])
    x = inp[:-1]
    y = inp[1:]
    y_mask = inp_mask[1:]
  # Define input embedding layer
  _i_embed = LinearCell(FLAGS.n_input_embed, bias=False,
                        num_inputs=FLAGS.n_class, input_is_int=True,
                        use_layer_norm=FLAGS.use_layer_norm)
  # Call input embedding layer
  h_i_emb_2d = _i_embed(x, 'i_embed')
  # Reshape into [max_seq_len-1, batch_size, num_units]
  h_i_emb_3d = tf.reshape(h_i_emb_2d,
                          [FLAGS.max_seq_len-1, -1, FLAGS.n_input_embed])
  # Define LSTM module
  _rnn = LSTMModule(FLAGS.n_hidden, use_layer_norm=FLAGS.use_layer_norm)
  # Call LSTM module
  h_rnn_3d, last_state = _rnn(h_i_emb_3d, state)
  # Reshape into [(max_seq_len-1)*batch_size, num_units]
  h_rnn_2d = tf.reshape(h_rnn_3d, [-1, FLAGS.n_hidden])
  # Define output layer
  _output = LinearCell(FLAGS.n_class, use_layer_norm=FLAGS.use_layer_norm)
  # Call output layer
  h_logits = _output(h_o_embed, 'output')
  # Transform labels into one-hot vectors
  y_1hot = tf.one_hot(tf.reshape(y, [-1]), depth=FLAGS.n_class)
  # Define loss and optimizer
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_1hot,
                                                          logits=h_logits)
  # Reshape into [max_seq_len-1, num_units]
  cross_entropy = tf.reshape(cross_entropy, [FLAGS.max_seq_len-1, -1])
  cost = tf.reduce_sum((cross_entropy * y_mask), reduction_indices=0)
  return cost, inp, inp_mask, state, last_state


def initial_states(FLAGS):
  init_state = np.zeros([2, FLAGS.batch_size, FLAGS.n_hidden], dtype=np.float32)
  return init_state


def sess_wrapper(x, x_mask, t_step, step_tr_cost, sess, G, recursive_states):
  _tr_cost, _, _state, _step_tr_cost = \
      sess.run([G[0], t_step, G[4], step_tr_cost],
               feed_dict={G[1]: x, G[2]: x_mask,
                          G[3]: recursive_states})
  return _tr_cost, _step_tr_cost, _state


def monitor(G, sess, train_set, valid_set, init_state, FLAGS,
            summary, S, summary_writer, _epoch, monitor_ops=None):
  # Process training set
  if train_set is None:
    tr_nats = 0
    tr_bits = 0
  else:
    _step = 0
    _cost = 0
    _len = 0
    _state = init_state
    for x in train_set:
      x, x_mask = gen_mask(x, FLAGS.max_seq_len)
      _tr_cost, _state = sess.run([G[0], G[4]],
                    feed_dict={G[1]: x, G[2]: x_mask, G[3]: _state})
      _cost += _tr_cost.sum()
      _len += (x_mask[1:]).sum()
    tr_nats = _cost / _len
    tr_bits = nats2bits(tr_nats)
  # Process validation set
  _step = 0
  _cost = 0
  _len = 0
  _state = init_state
  for x in valid_set:
    x, x_mask = gen_mask(x, FLAGS.max_seq_len)
    _val_cost, _state = sess.run([G[0], G[4]],
                  feed_dict={G[1]: x, G[2]: x_mask, G[3]: _state})
    _cost += _val_cost.sum()
    _len += (x_mask[1:]).sum()
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
