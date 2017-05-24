'''A hierarchical multiscale long short-term memory (HM-LSTM)
implementation using TensorFlow library.
'''
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import insert_item2dict
from mixer import nats2bits
from model import HMLSTMModule
from model import LinearCell
from ptb_train import TrainModel
from tensorflow.python.ops.math_ops import sigmoid


FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay factor')
flags.DEFINE_integer('batch_size', 64, 'Size of mini-batch')
flags.DEFINE_integer('n_epoch', 500, 'Maximum number of epochs')
flags.DEFINE_integer('display_freq', 100, 'Display frequency')
flags.DEFINE_integer('max_seq_len', 100, 'Maximum length of sequences')
flags.DEFINE_integer('n_hidden', 512, 'Number of RNN hidden units')
flags.DEFINE_integer('n_class', 50, 'Number of target symbols')
flags.DEFINE_integer('n_input_embed', 128, 'Number of input embedding dimension')
flags.DEFINE_integer('n_output_embed', 512, 'Number of output embedding dimension')
flags.DEFINE_integer('base_seed', 20170317, 'Base random seed')
flags.DEFINE_integer('add_seed', 0, 'Add this amount to the base random seed')
flags.DEFINE_boolean('start_from_ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_boolean('grad_clip', True, 'If true, clip the gradients')
flags.DEFINE_boolean('eval_train', False, 'If true, evaluate on train set')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_string('log_dir', '/raid/chungjun/nips2017/testbed/ptb_hmlstm',
                    'Directory path to files')
flags.DEFINE_integer('inf_seq_len', 2000, 'Inference sequence')
flags.DEFINE_integer('inf_im_scale', 10, 'Upsample the matrix size')
flags.DEFINE_string('use_impl_type', 'base', 'type of HM-LSTM implementation '
                    'could be one of {"base", "test"}')


def build_graph(FLAGS):
  """Define training graph"""
  with tf.device(FLAGS.device):
    # Graph input
    inp = tf.placeholder(tf.int32, [FLAGS.max_seq_len, None])
    inp_mask = tf.placeholder(tf.float32, [FLAGS.max_seq_len, None])
    state = tf.placeholder(tf.float32, [3, 2, None, FLAGS.n_hidden])
    boundary = tf.placeholder(tf.float32, [2, None, 1])
    x = inp[:-1]
    y = inp[1:]
    y_mask = inp_mask[1:]
  # Define input embedding layer
  _i_embed = LinearCell(FLAGS.n_input_embed, bias=False,
                        num_inputs=FLAGS.n_class, input_is_int=True)
  # Call input embedding layer
  h_i_emb_2d = _i_embed(x, 'i_embed')
  # Reshape into [max_seq_len-1, batch_size, num_units]
  h_i_emb_3d = tf.reshape(h_i_emb_2d,
                          [FLAGS.max_seq_len-1, -1, FLAGS.n_input_embed])
  # Define HM-LSTM module
  _rnn = HMLSTMModule(FLAGS.n_hidden, use_impl_type=FLAGS.use_impl_type)
  # Call HM-LSTM module
  (h_rnn_1_3d, h_rnn_2_3d, h_rnn_3_3d, z_1_3d, z_2_3d), \
      last_state, last_boundary = \
          _rnn(h_i_emb_3d, state, boundary)
  # Reshape into [(max_seq_len-1)*batch_size, num_units]
  h_rnn_1_2d = tf.reshape(h_rnn_1_3d, [-1, FLAGS.n_hidden])
  h_rnn_2_2d = tf.reshape(h_rnn_2_3d, [-1, FLAGS.n_hidden])
  h_rnn_3_2d = tf.reshape(h_rnn_3_3d, [-1, FLAGS.n_hidden])
  # Define output gating layer
  _o_gate = LinearCell(3, activation=sigmoid)
  # Call output gating layer
  h_o_gate = _o_gate([h_rnn_1_2d, h_rnn_2_2d, h_rnn_3_2d], 'o_gate')
  # Define output embedding layer
  _o_embed = LinearCell(FLAGS.n_output_embed, activation=tf.nn.relu)
  # Call output embedding layer
  h_o_embed= _o_embed([h_rnn_1_2d * tf.expand_dims(h_o_gate[:, 0], axis=1),
                       h_rnn_2_2d * tf.expand_dims(h_o_gate[:, 1], axis=1),
                       h_rnn_3_2d * tf.expand_dims(h_o_gate[:, 2], axis=1)],
                      'o_embed')
  # Define output layer
  _output = LinearCell(FLAGS.n_class)
  # Call output layer
  h_logits = _output(h_o_embed, 'output')
  # Transform labels into one-hot vectors
  y_1hot = tf.one_hot(tf.reshape(y, [-1]), depth=FLAGS.n_class)
  # Define loss and optimizer
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_1hot,
                                                          logits=h_logits)
  # Reshape into [max_seq_len-1, num_units]
  cross_entropy = tf.reshape(cross_entropy, [FLAGS.max_seq_len-1, -1])
  cost = tf.reduce_sum((cross_entropy * y_mask), axis=0)
  """Define inference graph with predefined sequence length"""
  with tf.device(FLAGS.device):
    test_inp = tf.placeholder(tf.int32, [FLAGS.inf_seq_len, None])
  # Call input embedding layer
  _h_i_emb_2d = _i_embed(test_inp, 'i_embed')
  # Reshape into [inf_seq_len, batch_size, num_units]
  _h_i_emb_3d = tf.reshape(_h_i_emb_2d,
                           [FLAGS.inf_seq_len, -1, FLAGS.n_input_embed])
  # Call LSTM module
  (_h_rnn_1_3d, _h_rnn_2_3d, _h_rnn_3_3d, _z_1_2d, _z_2_2d), _, _ = \
          _rnn(_h_i_emb_3d, state, boundary)
  # Process z and h
  _z_1_1d = tf.reshape(_z_1_2d, [FLAGS.inf_seq_len])
  _z_2_1d = tf.reshape(_z_2_2d, [FLAGS.inf_seq_len])
  z_stacked = tf.stack([_z_1_1d, _z_2_1d], axis=0)
  # Normalize by layers
  _h_1 = rescale_to_01(_h_rnn_1_3d, FLAGS.inf_seq_len)
  _h_2 = rescale_to_01(_h_rnn_2_3d, FLAGS.inf_seq_len)
  _h_3 = rescale_to_01(_h_rnn_3_3d, FLAGS.inf_seq_len)
  h_stacked = tf.stack([_h_1, _h_2, _h_3], axis=0)
  z_h_stacked = tf.stack([_h_1, _z_1_1d, _h_2, _z_2_1d, _h_3], axis=0)
  return cost, inp, inp_mask, state, boundary, last_state, last_boundary, \
          test_inp, z_stacked, h_stacked, z_h_stacked


def rescale_to_01(hidden, length):
  reshaped_hidden = tf.reshape(tf.norm(hidden, axis=2), [length])
  _min = tf.reduce_min(reshaped_hidden)
  _max = tf.reduce_max(reshaped_hidden)
  _range = _max - _min
  return (reshaped_hidden - _min) / (_range + 1e-8)


def initial_states(FLAGS):
  init_state = np.zeros([3, 2, FLAGS.batch_size, FLAGS.n_hidden],
                        dtype=np.float32)
  init_boundary = np.zeros([2, FLAGS.batch_size, 1], dtype=np.float32)
  return (init_state, init_boundary)


def sess_wrapper(x, x_mask, t_step, step_tr_cost, sess, G, recursive_states):
  _tr_cost, _, _state, _boundary, _step_tr_cost = \
          sess.run([G[0], t_step, G[5], G[6], step_tr_cost],
                   feed_dict={G[1]: x, G[2]: x_mask,
                              G[3]: recursive_states[0],
                              G[4]: recursive_states[1]})
  return _tr_cost, _step_tr_cost, (_state, _boundary)


def monitor(G, sess, train_set, valid_set, init_state, FLAGS,
            summary, S, summary_writer, _epoch,
            (z_op, z_state, h_op, h_state, z_h_op, z_h_state)):
  # Process training set
  if train_set is None:
    tr_nats = 0.0
    tr_bits = 0.0
  else:
    _cost = 0
    _len = 0
    _state = init_state[0]
    _boundary = init_state[1]
    for x in train_set:
      x, x_mask = gen_mask(x, FLAGS.max_seq_len)
      _tr_cost, _state, _boundary = sess.run([G[0], G[5], G[6]],
                    feed_dict={G[1]: x, G[2]: x_mask, G[3]: _state,
                               G[4]: _boundary})
      _cost += _tr_cost.sum()
      _len += (x_mask[1:]).sum()
    tr_nats = _cost / _len
    tr_bits = nats2bits(tr_nats)
  # Process validation set
  _cost = 0
  _len = 0
  _state = init_state[0]
  _boundary = init_state[1]
  for x in valid_set:
    x, x_mask = gen_mask(x, FLAGS.max_seq_len)
    _val_cost, _state, _boundary = sess.run([G[0], G[5], G[6]],
                  feed_dict={G[1]: x, G[2]: x_mask, G[3]: _state,
                             G[4]: _boundary})
    _cost += _val_cost.sum()
    _len += (x_mask[1:]).sum()
  val_nats = _cost / _len
  val_bits = nats2bits(val_nats)
  # Write boundary pattern as an image
  x = np.expand_dims(valid_set.data[:FLAGS.inf_seq_len], axis=1)
  x_mask = np.ones(x.shape).astype('float32')
  init_state = np.zeros([3, 2, 1, FLAGS.n_hidden],
                        dtype=np.float32)
  init_boundary = np.zeros([2, 1, 1], dtype=np.float32)
  z_stacked, h_stacked, z_h_stacked = sess.run([G[8], G[9], G[10]],
                                               feed_dict={G[7]: x,
                                                          G[3]: init_state,
                                                          G[4]: init_boundary})
  # Define canvases
  z_rate = np.mean(z_stacked.squeeze(), axis=1)
  print("First z activation rate: %f, \nSecond z activation rate: %f"
        % (z_rate[0], z_rate[1]))
  z_canvas = np.zeros([2, FLAGS.inf_seq_len, 1])
  h_canvas = np.zeros([3, FLAGS.inf_seq_len, 1])
  z_h_canvas = np.zeros([5, FLAGS.inf_seq_len, 1])
  # Fill canvases
  z_canvas[:, :, 0] = z_stacked[::-1]
  h_canvas[:, :, 0] = h_stacked[::-1]
  z_h_canvas[:, :, 0] = z_h_stacked[::-1]
  # Upscale canveses
  z_canvas = upsample_matrix(z_canvas)
  h_canvas = upsample_matrix(h_canvas)
  z_h_canvas = upsample_matrix(z_h_canvas)
  # Run session
  z_canvas = np.expand_dims(z_canvas, 0)
  h_canvas = np.expand_dims(h_canvas, 0)
  z_h_canvas = np.expand_dims(z_h_canvas, 0)
  # Write summary logs
  _z_state, _h_state, _z_h_state, \
  _epoch_tr_bits, _epoch_val_bits, _z1_active_rate, _z2_active_rate = \
            sess.run([z_op, h_op, z_h_op, S[0][1], S[0][2], S[0][3], S[0][4]],
                     feed_dict={z_state: z_canvas,
                                h_state: h_canvas,
                                z_h_state: z_h_canvas,
                                S[1][1]: tr_bits,
                                S[1][2]: val_bits,
                                S[1][3]: z_rate[0],
                                S[1][4]: z_rate[1]})
  if train_set is not None:
    summary_writer.add_summary(_epoch_tr_bits, _epoch)
  summary_writer.add_summary(_epoch_val_bits, _epoch)
  summary_writer.add_summary(_z1_active_rate, _epoch)
  summary_writer.add_summary(_z2_active_rate, _epoch)
  summary_writer.add_summary(_z_state, _epoch)
  summary_writer.add_summary(_h_state, _epoch)
  summary_writer.add_summary(_z_h_state, _epoch)
  summary_writer.flush()
  insert_item2dict(summary, 'z_1_rate', z_rate[0])
  insert_item2dict(summary, 'z_2_rate', z_rate[1])
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
    z1_rate = tf.placeholder(tf.float32, [])
    z2_rate = tf.placeholder(tf.float32, [])
    best_epoch_val_bits = tf.summary.scalar("best_valid_bits", best_val_bits)
    epoch_tr_bits = tf.summary.scalar("train_bits", tr_bits)
    epoch_val_bits = tf.summary.scalar("valid_bits", val_bits)
    z1_active_rate = tf.summary.scalar("z1_active_rate", z1_rate)
    z2_active_rate = tf.summary.scalar("z2_active_rate", z2_rate)
  return ([best_epoch_val_bits, epoch_tr_bits, epoch_val_bits, z1_active_rate,
           z2_active_rate],
          [best_val_bits, tr_bits, val_bits, z1_rate, z2_rate])


def add_monitor_op(FLAGS):
  z_state = tf.placeholder(tf.float32, shape=[None,
                                              2 * FLAGS.inf_im_scale,
                                              FLAGS.inf_seq_len *
                                              FLAGS.inf_im_scale,
                                              1], name='img_tensor_1')
  h_state = tf.placeholder(tf.float32, shape=[None,
                                              3 * FLAGS.inf_im_scale,
                                              FLAGS.inf_seq_len *
                                              FLAGS.inf_im_scale,
                                              1], name='img_tensor_2')
  z_h_state = tf.placeholder(tf.float32, shape=[None,
                                                5 * FLAGS.inf_im_scale,
                                                FLAGS.inf_seq_len *
                                                FLAGS.inf_im_scale,
                                                1], name='img_tensor_3')
  z_op = tf.summary.image('z_state', z_state)
  h_op = tf.summary.image('h_state', h_state)
  z_h_op = tf.summary.image('z_h_state', z_h_state)
  return (z_op, z_state, h_op, h_state, z_h_op, z_h_state)


def upsample_matrix(small_matrix, scale=10):
  height, width, n_channel = small_matrix.shape
  big_matrix = np.zeros([height * scale, width * scale, n_channel])
  for i in xrange(n_channel):
      big_matrix[:, :, i] = \
              small_matrix[:, :, i].repeat(scale, axis=0).repeat(scale, axis=1)
  return big_matrix


def main(_):
  if not FLAGS.start_from_ckpt:
    if tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
  train = TrainModel(FLAGS, build_graph, monitor, initial_states, sess_wrapper,
                     define_summary, add_monitor_op)
  train()


if __name__ == "__main__":
  tf.app.run()
  main(_)
