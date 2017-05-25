'''A Long short-term memory (LSTM) with skimming implementation using TensorFlow library.
'''
import numpy as np
import tensorflow as tf

from mixer import gen_mask
from mixer import nats2bits
from mixer import insert_item2dict
from model import LinearCell
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
flags.DEFINE_integer('n_action', 3, 'Number of actions (max skim size)')
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

TrainGraph = namedtuple('TrainGraph', 'ml_cost rl_cost seq_x_data seq_x_mask seq_y_data init_state seq_action seq_advantage')
SampleGraph = namedtuple('SampleGraph', 'action_probs label_probs c_state h_state x_data prev_c_state prev_h_state')

def build_graph(FLAGS):
  ##########################
  # Define input variables #
  ##########################
  with tf.device(FLAGS.device):
    ################
    # for training #
    ################
    # input sequence (batch_size, seq_len, feat_size)
    seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_input))
    # input mask  (batch_size, seq_len)
    seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
    # target label (batch_size, seq_len)
    seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))
    # init states (batch_size, 2, num_hiddens)
    init_state = tf.placeholder(tf.float32, shape=(None, 2, FLAGS.n_hidden))
    # action_seq (batch_size, seq_len)
    seq_action = tf.placeholder(tf.int32, shape=(None, None))
    # action advantage (reward-baseline) (batch_size, seq_len)
    seq_advantage = tf.placeholder(tf.float32, shape=(None, None))

    #####################
    # for action sample #
    #####################
    # input data (batch_size, feat_size)
    step_x_data = tf.placeholder(tf.float32, shape=(None, FLAGS.n_input))
    # previous state (batch_size, 2, num_hiddens)
    prev_states = tf.placeholder(tf.float32, shape=(None, 2, FLAGS.n_hidden))

  ######################
  # Define LSTM module #
  ######################
  with tf.variable_scope('rnn'):
    _rnn = LSTMModule(num_units=FLAGS.n_hidden)

  ##########################
  # Define labeling module #
  ##########################
  with tf.variable_scope('label'):
    _label_logit = LinearCell(num_units=FLAGS.n_class)

  #############################
  # Define skim action module #
  #############################
  with tf.variable_scope('action'):
    _action_logit = LinearCell(num_units=FLAGS.n_action)

  ################
  # For training #
  ################
  # Run rnn
  seq_hid_3d, _ = _rnn(seq_x_data, init_state)
  seq_hid_2d = tf.reshape(seq_hid_3d, [-1, FLAGS.n_hidden])

  # Compute label logits for ML
  seq_label_logits = _label_logit(seq_hid_2d, 'label_logit')

  # Compute action probs for RL
  seq_action_logits = _action_logit(seq_hid_2d, 'action_logit')
  seq_action_probs = tf.nn.softmax(seq_action_logits) + 1e-8

  # Compute ML_cost
  ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, labels=seq_y_data)
  ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))/tf.reduce_sum(tf.reshape(seq_x_mask, [-1]))

  # Compute RL_cost
  rl_cost = tf.reduce_sum(tf.log(seq_action_probs)*tf.one_hot(seq_action, depth=FLAGS.n_action), axis=-1)
  rl_cost *= seq_advantage
  rl_cost = tf.reduce_sum(rl_cost*tf.reshape(seq_x_mask, [-1]))/tf.reduce_sum(tf.reshape(seq_x_mask, [-1]))

  #####################
  # For action sample #
  #####################
  # Run rnn
  step_c_state, step_h_state = _rnn(step_x_data, prev_states, one_step=True)

  # Compute label probs
  step_label_logits = _label_logit(step_h_state, 'label_logit')
  step_label_probs = tf.nn.softmax(logits=step_label_logits)

  # Compute action probs
  step_action_logits = _action_logit(step_h_state, 'action_logit')
  step_action_probs = tf.nn.softmax(logits=step_action_logits)


  ###############
  # Train graph #
  ###############
  train_graph = TrainGraph(ml_cost,
                           rl_cost,
                           seq_x_data,
                           seq_x_mask,
                           seq_y_data,
                           init_state,
                           seq_action, seq_advantage)

  ################
  # Sample graph #
  ################
  sample_graph = SampleGraph(step_c_state,
                             step_h_state,
                             step_label_probs,
                             step_action_probs,
                             step_x_data,
                             prev_states)

  return train_graph, sample_graph
