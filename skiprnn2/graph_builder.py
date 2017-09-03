import numpy as np

from collections import namedtuple

from mixer import categorical_ent
from model import LinearCell
import utils

import tensorflow as tf 

def stop_gradient(tensor, args):
    if args.no_stop_gradient:
        return tensor
    else:
        return tf.stop_gradient(tensor)

def lstm_state(n_hidden, layer, n_proj, backward=False):
    c_name = 'cstate_{}'.format(layer)
    if backward:
        c_name = '{}_bw'.format(c_name)

    h_name = 'hstate_{}'.format(layer)
    if backward:
        h_name = '{}_bw'.format(h_name)

    c = tf.placeholder(tf.float32, shape=(None, n_hidden), name=c_name)
    h = tf.placeholder(tf.float32, shape=(None, n_hidden if n_proj == 0 else n_proj), name=h_name)

    return tf.contrib.rnn.LSTMStateTuple(c, h)

def lstm_init_state(args, backward=False):
    return tuple( lstm_state(args.n_hidden, l, args.n_proj, backward) for l in range(args.n_layer))

def match_c(opname):
    return 'rnn/multi_rnn_cell/cell' in opname and opname.endswith('lstm_cell/add_1')

def match_h(opname):
    return 'rnn/multi_rnn_cell/cell' in opname and opname.endswith('lstm_cell/mul_2')

def match_c_fw(opname):
    return 'bidirectional_rnn/fw' in opname and opname.endswith('Exit_2')

def match_h_fw(opname):
    return 'bidirectional_rnn/fw' in opname and opname.endswith('Exit_3')

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
        'step_action_samples', 'step_x_data', 'init_state', 'action_entropy', 'step_pred_idx']

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
    step_label_logits = tf.layers.dense(step_h_state, args.n_class, name='label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    step_pred_idx = tf.argmax(step_label_logits, axis=1)

    step_action_logits = tf.layers.dense(step_h_state, args.n_action, name='action_logit')

    step_action_probs = tf.nn.softmax(logits=step_action_logits, name='step_action_probs')
    step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1, name='step_action_samples')
    step_action_entropy = categorical_ent(step_action_probs)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
    seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

    seq_label_logits = tf.layers.dense(seq_hid_2d, args.n_class, reuse=True, name='label_logit')

    y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=args.n_class)

    ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
        labels=y_1hot)
    ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

    pred_idx = tf.argmax(seq_label_logits, axis=1)

    seq_hid_3d_rl = seq_hid_3d[:,:-1,:] 
    seq_hid_2d_rl = tf.reshape(seq_hid_3d_rl, [-1, args.n_hidden])
    seq_hid_2d_rl = stop_gradient(seq_hid_2d_rl, args)

    seq_action_logits = tf.layers.dense(seq_hid_2d_rl, args.n_action, name='action_logit', reuse=True)
        
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
        step_action_probs, step_action_samples, step_x_data, init_state, step_action_entropy, step_pred_idx)

    return train_graph, sample_graph

def build_graph_sv(args):
    TrainGraph = namedtuple('TrainGraph', 
        'ml_cost rl_cost seq_x_data seq_x_mask seq_y_data seq_jump_data init_state pred_idx seq_action_samples')
    TestGraph = namedtuple('TestGraph', 
        'step_h_state step_last_state step_label_probs step_action_probs step_pred_idx step_x_data init_state step_action_samples')

    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input))
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None))
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state = lstm_init_state(args)

        seq_jump_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(args) for _ in range(args.n_layer)])

    # testing graph (step_h_state == step_last_state)
    step_h_state, step_last_state = cell(step_x_data, init_state, scope='rnn/multi_rnn_cell')

    step_label_logits = tf.layers.dense(step_h_state, args.n_class, name='label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    if args.use_unimodal: 
        scalar_outputs = tf.layers.dense(step_h_state, 1, name='action_logit', activation=tf.nn.softplus)
        poi = tf.contrib.distributions.Poisson(scalar_outputs)
        k_values = list(np.arange(0.0, args.n_action, 1.0))
        step_action_logits = poi.prob(k_values)/args.tau # smoothing
        step_action_probs = tf.nn.softmax(logits=step_action_logits, name='step_action_probs')
        step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1, name='step_action_samples')
    else:
        step_action_logits = tf.layers.dense(step_h_state, args.n_action, name='action_logit')
        step_action_probs = tf.nn.softmax(logits=step_action_logits, name='step_action_probs')
        step_action_samples = tf.multinomial(logits=step_action_logits, num_samples=1, name='step_action_samples')

    step_pred_idx = tf.argmax(step_action_logits, axis=1, name='step_pred_idx')
    
    test_graph = TestGraph(step_h_state, step_last_state, step_label_probs, 
        step_action_probs, step_pred_idx, step_x_data, init_state, step_action_samples)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
    seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

    seq_label_logits = tf.layers.dense(seq_hid_2d, args.n_class, name='label_logit', reuse=True)

    y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=args.n_class)

    ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
        labels=y_1hot)

    ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

    pred_idx = tf.argmax(seq_label_logits, axis=1)

    seq_hid_3d_rl = seq_hid_3d[:,:-1,:]
    seq_hid_2d_rl = tf.reshape(seq_hid_3d_rl, [-1, args.n_hidden])
    seq_hid_2d_rl = stop_gradient(seq_hid_2d_rl, args)

    if args.use_unimodal: 
        seq_scalar_outputs = tf.layers.dense(seq_hid_2d_rl, 1, name='action_logit', activation=tf.nn.softplus, reuse=True)
        poi = tf.contrib.distributions.Poisson(seq_scalar_outputs)
        k_values = list(np.arange(0.0, args.n_action, 1.0))
        seq_action_logits = poi.prob(k_values)/args.tau # smoothing
        seq_action_probs = tf.nn.softmax(logits=seq_action_logits, name='seq_action_probs')
        seq_action_samples = tf.multinomial(logits=seq_action_logits, num_samples=1, name='seq_action_samples')
    else:
        seq_action_logits = tf.layers.dense(seq_hid_2d_rl, args.n_action, name='action_logit', reuse=True)
        seq_action_probs = tf.nn.softmax(seq_action_logits)
        seq_action_samples = tf.multinomial(logits=seq_action_logits, num_samples=1, name='seq_action_samples')

    jump_1hot = tf.one_hot(tf.reshape(seq_jump_data, [-1]), depth=args.n_action)

    rl_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_action_logits, labels=jump_1hot)
    rl_cost = tf.reduce_sum(rl_cost*tf.reshape(seq_x_mask[:,:-1], [-1]))

    train_graph = TrainGraph(ml_cost, rl_cost, seq_x_data, seq_x_mask, 
        seq_y_data, seq_jump_data, init_state, pred_idx, seq_action_samples)

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

    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(args) for _ in range(args.n_layer)])

    # testing graph
    step_h_state, step_last_state = cell(step_x_data, init_state, scope='rnn/multi_rnn_cell')

    step_label_logits = tf.layers.dense(step_h_state, args.n_class)
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    test_graph = TestGraph(step_x_data, init_state, step_last_state, step_label_probs)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
    seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden if args.n_proj == 0 else args.n_proj])

    seq_label_logits = tf.layers.dense(seq_hid_2d, args.n_class, reuse=True)   

    y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=args.n_class)

    ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
        labels=y_1hot)
    ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

    pred_idx = tf.argmax(seq_label_logits, axis=1)

    seq_label_probs = tf.nn.softmax(seq_label_logits, name='seq_label_probs')

    train_graph = TrainGraph(ml_cost, seq_x_data, seq_x_mask, seq_y_data, init_state, pred_idx, seq_label_probs)

    return train_graph, test_graph

def build_graph_subsample_tbptt(args):
    TrainGraph = namedtuple('TrainGraph', 
        'ml_cost seq_x_data seq_x_mask seq_y_data ' + \
        'init_state_fw init_state_bw pred_idx seq_label_probs ' + \
        'output_state_fw output_state_bw outputs')
    
    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input), name='seq_x_data')
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name='seq_x_mask')
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state_fw = lstm_init_state(args, backward=False)
        init_state_bw = lstm_init_state(args, backward=True)

    cells_fw = [lstm_cell(args) for _ in range(args.n_layer)]
    cells_bw = [lstm_cell(args) for _ in range(args.n_layer)]

    seq_lengths = tf.reduce_sum(tf.to_int32(seq_x_mask), 1)
    outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw, cells_bw, seq_x_data, init_state_fw, init_state_bw, sequence_length=seq_lengths)
    # outputs: n_batch, n_seq, n_hidden * 2

    seq_hid_2d = tf.reshape(outputs, [-1, args.n_hidden * 2 if args.n_proj == 0 else args.n_proj * 2])

    seq_label_logits = tf.layers.dense(seq_hid_2d, args.n_class, name='label_logit')   
    y_1hot = tf.one_hot(tf.reshape(seq_y_data, [-1]), depth=args.n_class)

    ml_cost = tf.nn.softmax_cross_entropy_with_logits(logits=seq_label_logits, 
        labels=y_1hot)
    ml_cost = tf.reduce_sum(ml_cost*tf.reshape(seq_x_mask, [-1]))

    pred_idx = tf.argmax(seq_label_logits, axis=1)

    seq_label_probs = tf.nn.softmax(seq_label_logits, name='seq_label_probs')

    train_graph = TrainGraph(ml_cost, seq_x_data, seq_x_mask, seq_y_data, init_state_fw, init_state_bw, 
        pred_idx, seq_label_probs, output_state_fw, output_state_bw, outputs)

    return train_graph

