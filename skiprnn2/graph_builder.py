import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple

from mixer import gen_mask
from mixer import insert_item2dict
from mixer import save_npz2
from mixer import gen_episode_with_seg_reward
from mixer import LinearVF, compute_advantage2
from mixer import categorical_ent, expand_output
from mixer import lstm_state, gen_zero_state, feed_init_state
from model import LinearCell

def build_graph_ri(args):
    tg_fields = ['ml_cost', 'rl_cost', 'seq_x_data', 'seq_x_mask',
        'seq_y_data', 'seq_y_data_for_action', 'init_state', 'seq_action', 'seq_advantage', 'seq_action_mask', 'pred_idx']

    sg_fields = ['step_h_state', 'step_last_state', 'step_label_probs', 'step_action_probs',
        'step_action_samples', 'step_x_data', 'step_y_data_for_action', 'init_state', 'action_entropy', 'sample_y']

    TrainGraph = namedtuple('TrainGraph', ' '.join(tg_fields))
    SampleGraph = namedtuple('SampleGraph', ' '.join(sg_fields))
    
    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(tf.float32, shape=(None, None, args.n_input))
        seq_x_mask = tf.placeholder(tf.float32, shape=(None, None))
        seq_y_data = tf.placeholder(tf.int32, shape=(None, None))
        seq_y_data_for_action = tf.placeholder(tf.int32, shape=(None,None))
        
        init_state = tuple( lstm_state(args.n_hidden, l) for l in range(args.n_layer))

        seq_action = tf.placeholder(tf.float32, shape=(None, None, args.n_action))
        seq_advantage = tf.placeholder(tf.float32, shape=(None, None))
        seq_action_mask = tf.placeholder(tf.float32, shape=(None, None))
        
        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

        embedding = tf.get_variable("embedding", [args.n_class, args.n_embedding], dtype=tf.float32)
        step_y_data_for_action = tf.placeholder(tf.int32, shape=(None,), name='step_y_data_for_action')
        seq_y_input = tf.nn.embedding_lookup(embedding, seq_y_data_for_action)

        sample_y = tf.placeholder(tf.bool, name='sample_y')

    def lstm_cell():
        return tf.contrib.rnn.LSTMCell(num_units=args.n_hidden, forget_bias=0.0)
       
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(args.n_layer)])

    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class)

    with tf.variable_scope('action'):
        _action_logit = LinearCell(num_units=args.n_action)

    # sampling graph
    step_h_state, step_last_state = cell(step_x_data, init_state, scope='rnn/multi_rnn_cell')

    # no need to do stop_gradient because training is not done for the sampling graph
    step_label_logits = _label_logit(step_h_state, 'label_logit')
    step_label_probs = tf.nn.softmax(logits=step_label_logits, name='step_label_probs')

    step_y_input_answer = tf.nn.embedding_lookup(embedding, step_y_data_for_action)
    step_y_1hot_pred = tf.argmax(step_label_probs, axis=-1)
    step_y_input_pred = tf.nn.embedding_lookup(embedding, step_y_1hot_pred)
    step_y_input = tf.where(sample_y, step_y_input_pred, step_y_input_answer)

    if args.n_embedding > 0:
        step_action_logits = _action_logit([step_h_state, step_y_input], 'action_logit')
    else:
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

    seq_y_input_2d = tf.reshape(seq_y_input[:,:-1:], [-1, args.n_embedding])

    if args.n_embedding > 0:
        seq_action_logits = _action_logit([seq_hid_2d_rl, seq_y_input_2d], 'action_logit')
    else:
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
        seq_y_data, seq_y_data_for_action, init_state, seq_action, seq_advantage, seq_action_mask, pred_idx)

    sample_graph = SampleGraph(step_h_state, step_last_state, step_label_probs,
        step_action_probs, step_action_samples, step_x_data, step_y_data_for_action, init_state, step_action_entropy, sample_y)

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


def build_graph_subsample(args):

    TrainGraph = namedtuple('TrainGraph', 'ml_cost seq_x_data seq_x_mask seq_y_data init_state pred_idx seq_label_probs')
    TestGraph = namedtuple('TestGraph', 'step_x_data init_state step_last_state step_label_probs')

    with tf.device(args.device):
        # n_batch, n_seq, n_feat
        seq_x_data = tf.placeholder(dtype=tf.float32, shape=(None, None, args.n_input), name='seq_x_data')
        seq_x_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name='seq_x_mask')
        seq_y_data = tf.placeholder(dtype=tf.int32, shape=(None, None))

        init_state = tuple( lstm_state(args.n_hidden, l) for l in range(args.n_layer))

        step_x_data = tf.placeholder(tf.float32, shape=(None, args.n_input), name='step_x_data')

    with tf.variable_scope('rnn'):
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(num_units=args.n_hidden, forget_bias=0.0)
           
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(args.n_layer)])

    with tf.variable_scope('label'):
        _label_logit = LinearCell(num_units=args.n_class)

    # training graph
    seq_hid_3d, _ = tf.nn.dynamic_rnn(cell=cell, inputs=seq_x_data, initial_state=init_state, scope='rnn')
    seq_hid_2d = tf.reshape(seq_hid_3d, [-1, args.n_hidden])

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

    train_graph = TrainGraph(ml_cost,
                                                     seq_x_data,
                                                     seq_x_mask,
                                                     seq_y_data,
                                                     init_state,
                                                     pred_idx,
                                                     seq_label_probs)

    test_graph = TestGraph(step_x_data, init_state, step_last_state, step_label_probs)

    return train_graph, test_graph
