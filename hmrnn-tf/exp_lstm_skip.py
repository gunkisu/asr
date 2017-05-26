'''A Long short-term memory (LSTM) implementation using TensorFlow library.
'''
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import nats2bits
from mixer import insert_item2dict
from model import LinearCell
from model import LSTMModule
from train_loop_skip import TrainModel

from collections import namedtuple

from libs.utils import skip_frames_fixed

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
flags.DEFINE_integer('skip', 1, '')

Graph = namedtuple('Graph', 'cost x x_mask state last_state y')

def build_graph(FLAGS):
  """Define training graph.
  """
  with tf.device(FLAGS.device):
    # Graph input
    x = tf.placeholder(tf.float32, shape=(None, None, FLAGS.n_input)) # (seq_len, batch_size, n_input)
    x_mask = tf.placeholder(tf.float32, shape=(None, None)) # (seq_len, batch_size)
    state = tf.placeholder(tf.float32, shape=(2, None, FLAGS.n_hidden))
    y = tf.placeholder(tf.int32, shape=(None, None)) # (seq_len, batch_size)

  # Define LSTM module
  _rnn = LSTMModule(FLAGS.n_hidden)
  # Call LSTM module
  h_rnn_3d, last_state = _rnn(x, state)
  # Reshape into [seq_len*batch_size, num_units]
  h_rnn_2d = tf.reshape(h_rnn_3d, [-1, FLAGS.n_hidden])
  # Define output layer
  _output = LinearCell(FLAGS.n_class)
  # Call output layer [seq_len*batch_size, n_class]
  h_logits = _output(h_rnn_2d, 'output')
  # Transform labels into one-hot vectors [seq_len*batch_size, n_class]
  y_1hot = tf.one_hot(tf.reshape(y, [-1]), depth=FLAGS.n_class)
  # Define loss and optimizer [seq_len*batch_size]
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_1hot,
                                                          logits=h_logits)
  # Reshape into [seq_len, batch_size]
#  cross_entropy = tf.reshape(cross_entropy, [-1, FLAGS.batch_size])
  cost = tf.reduce_sum((cross_entropy * tf.reshape(x_mask, [-1])), reduction_indices=0)
  return Graph(cost, x, x_mask, state, last_state, y)


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
      orig_x, orig_x_mask, _, _, orig_y, _ = batch
      
      for x, x_mask, y in skip_frames_fixed([orig_x, orig_x_mask, orig_y], FLAGS.skip+1, return_first=True):
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
    orig_x, orig_x_mask, _, _, orig_y, _ = batch
  
    for x, x_mask, y in skip_frames_fixed([orig_x, orig_x_mask, orig_y], FLAGS.skip+1, return_first=True):
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
