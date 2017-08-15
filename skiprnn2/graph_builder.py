import numpy as np
import tensorflow as tf

from collections import namedtuple

from mixer import categorical_ent
from model import LinearCell
import utils

def lstm_state(n_hidden, layer, n_proj):
    
    return tf.contrib.rnn.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, n_hidden), name='cstate_{}'.format(layer)), 
        tf.placeholder(tf.float32, shape=(None, n_hidden if n_proj == 0 else n_proj), name='hstate_{}'.format(layer)))

def lstm_init_state(args):
    return tuple( lstm_state(args.n_hidden, l, args.n_proj) for l in range(args.n_layer))

def match_c(opname):
    return 'rnn/multi_rnn_cell/cell' in opname and 'lstm_cell/add_1' in opname

def match_h(opname):
    return 'rnn/multi_rnn_cell/cell' in opname and 'lstm_cell/mul_2' in opname

def lstm_cell(args):
    if args.use_layer_norm: 
        return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=args.n_hidden, forget_bias=0.0)

    else:
        if args.n_proj > 0:
            return tf.contrib.rnn.LSTMCell(num_units=args.n_hidden, num_proj=args.n_proj, forget_bias=0.0)
        else:
            return tf.contrib.rnn.LSTMCell(num_units=args.n_hidden, forget_bias=0.0)

def build_graph_ri(args):
    tg_fields = ['ml_cost', 'rl_cost', 'seq_x_data', 'seq_x_mask',
        'seq_y_data', 'init_state', 'seq_action', 'seq_advantage', 'seq_action_mask', 'pred_idx']

    sg_fields = ['step_h_state', 'step_last_state', 'step_label_probs', 'step_action_probs',
        'step_action_samples', 'step_x_data', 'init_state', 'action_entropy']

    TrainGraph = namedtuple('TrainGraph', ' '.join(tg_fields))
    SampleGraph = namedtuple('SampleGraph', ' '.join(sg_fields))
    
    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(tf.float32, shape=(None, None, args.n_input))
        seq_x_mask = tf.placeholder(tf.float32, shape=(None, None))
        seq_y_data = tf.placeholder(tf.int32, shape=(None, None))
        
        init_state = lstm_init_state(args)

        seq_action = tf.placeholder(tf.float32, shape=(None, None, args.n_action))
        seq_advantage = tf.placeholder(tf.float32, shape=(None, None))
        seq_action_mask = tf.placeholder(tf.float32, shape=(None, None))
        
        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')
       
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(args) for _ in range(args.n_layer)])

    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class)

    with tf.variable_scope('action'):
        _action_logit = LinearCell(num_units=args.n_action)

    # sampling graph
    step_h_state, step_last_state = cell(step_x_data, init_state, scope='rnn/multi_rnn_cell')

    # no need to do stop_gradient because training is not done for the sampling graph
    step_label_logits = _label_logit(step_h_state, 'label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    step_action_logits = _action_logit(step_h_state, 'action_logit')

    step_action_probs = tf.nn.softmax(logits=step_action_logits, name='step_action_probs')
    step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1, name='step_action_samples')
    step_action_entropy = categorical_ent(step_action_probs)

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

    action_prob_entropy = categorical_ent(seq_action_probs)
    action_prob_entropy *= tf.reshape(seq_action_mask, [-1])
    action_prob_entropy = tf.reduce_sum(action_prob_entropy)/tf.reduce_sum(seq_action_mask)

    # Optimizing over the surrogate function 
    rl_cost = tf.reduce_sum(tf.log(seq_action_probs+1e-8) \
        * tf.reshape(seq_action, [-1,args.n_action]), axis=-1)
    rl_cost *= tf.reshape(seq_advantage, [-1])
    rl_cost = -tf.reduce_sum(rl_cost*tf.reshape(seq_action_mask, [-1]))

    train_graph = TrainGraph(ml_cost, rl_cost, seq_x_data, seq_x_mask, 
        seq_y_data, init_state, seq_action, seq_advantage, seq_action_mask, pred_idx)

    sample_graph = SampleGraph(step_h_state, step_last_state, step_label_probs,
        step_action_probs, step_action_samples, step_x_data, init_state, step_action_entropy)

    return train_graph, sample_graph

def build_graph_sv(args):
    TrainGraph = namedtuple('TrainGraph', 
        'ml_cost rl_cost seq_x_data seq_x_mask seq_y_data seq_jump_data init_state pred_idx')
    TestGraph = namedtuple('TestGraph', 
        'step_h_state step_last_state step_label_probs step_action_probs step_pred_idx step_x_data init_state')

    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input))
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state = lstm_init_state(args)

        seq_jump_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(args) for _ in range(args.n_layer)])

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


def build_graph_subsample(args):
    TrainGraph = namedtuple('TrainGraph', 'ml_cost seq_x_data seq_x_mask seq_y_data init_state pred_idx seq_label_probs')
    TestGraph = namedtuple('TestGraph', 'step_x_data init_state step_last_state step_label_probs')

    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input), name='seq_x_data')
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name='seq_x_mask')
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state = lstm_init_state(args)

        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

    with tf.variable_scope('rnn'):
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(args) for _ in range(args.n_layer)])

    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
    seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden if args.n_proj == 0 else args.n_proj])

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

    train_graph = TrainGraph(ml_cost, seq_x_data, seq_x_mask, seq_y_data, init_state, pred_idx, seq_label_probs)

    test_graph = TestGraph(step_x_data, init_state, step_last_state, step_label_probs)

    return train_graph, test_graph
