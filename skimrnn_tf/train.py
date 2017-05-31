import os
import numpy as np
import tensorflow as tf
from data.fuel_utils import create_ivector_datastream
from model import LinearCell, SkimLSTMModule
from collections import namedtuple

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model size
flags.DEFINE_integer('n_input', 123, 'Number of inputs')
flags.DEFINE_integer('n_hidden', 128, 'Number of LSTM hidden units')
flags.DEFINE_integer('n_class', 3436, 'Number of target symbols')
flags.DEFINE_integer('n_layer', 3, 'Number of layers')
flags.DEFINE_float('forget_bias', 1.0, 'forget bias')

# Action size
flags.DEFINE_integer('n_read', 1, 'Number of minimum read')
flags.DEFINE_integer('n_action', 5, 'Number of maximum skim')

# Random Seet
flags.DEFINE_integer('base_seed', 2222, 'Base random seed')
flags.DEFINE_integer('add_seed', 0, 'Add this amount to the base random seed')

# Learning
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay factor')
flags.DEFINE_float('ml_grad_clip', 1.0, 'Gradient norm clipping (ML)')
flags.DEFINE_float('rl_grad_clip', 0.0, 'Gradient norm clipping (RL)')
flags.DEFINE_float('ml_learning_rate', 0.001, 'Initial ml learning rate')
flags.DEFINE_float('rl_learning_rate', 0.01, 'Initial rl learning rate')
flags.DEFINE_integer('batch_size', 16, 'Size of mini-batch')
flags.DEFINE_integer('n_epoch', 200, 'Maximum number of epochs')

# Dataset
flags.DEFINE_string('data_path', '/u/songinch/song/data/speech/wsj_fbank123.h5', 'Data path')
flags.DEFINE_string('train_dataset', 'train_si284', 'Train set name')
flags.DEFINE_string('valid_dataset', 'test_dev93', 'Valid set name')
flags.DEFINE_string('test_dataset', 'test_eval92', 'Test set name')

# Output
flags.DEFINE_boolean('start_from_ckpt', False, 'If true, start from a ckpt')
flags.DEFINE_string('log_dir', 'skim_lstm', 'Directory path to files')
flags.DEFINE_integer('display_freq', 100, 'Display frequency')
flags.DEFINE_integer('evaluation_freq', 2000, 'Display frequency')

# Aux
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_boolean('no_copy', False, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')

graph_attr_list = ['x_data',
                   'x_mask',
                   'y_data',
                   'init_state',
                   'init_cntr',
                   'mean_accr',
                   'mean_ml_cost',
                   'mean_rl_cost',
                   'mean_bl_cost',
                   'ml_cost_param',
                   'rl_cost_param_list',
                   'bl_cost_param_list']

Graph = namedtuple('Graph', ' '.join(graph_attr_list))


def build_graph(FLAGS):
    # Define input data
    with tf.device(FLAGS.device):
        # input sequence
        x_data = tf.placeholder(dtype=tf.float32,
                                shape=(None, None, FLAGS.n_input),
                                name='x_data')

        # input mask
        x_mask = tf.placeholder(dtype=tf.float32,
                                shape=(None, None, 1),
                                name='x_mask')

        # gt label
        y_data = tf.placeholder(dtype=tf.int32,
                                shape=(None, None),
                                name='y_data')

        # init state (but mostly init with 0s)
        init_state = tf.placeholder(dtype=tf.float32,
                                    shape=(None, FLAGS.n_hidden),
                                    name='init_state')

        # init counter (but mostly init with 0s)
        init_cntr = tf.placeholder(dtype=tf.float32,
                                   shape=(None, 1),
                                   name='init_cntr')

    # Define model (get outputs from each layer and each direction)
    fwd_hid_list = []
    fwd_act_lgp_list = []
    fwd_act_mask_list = []

    bwd_hid_list = []
    bwd_act_lgp_list = []
    bwd_act_mask_list = []

    # For each layer
    prev_hid_data = x_data
    for l in range(FLAGS.n_layer):
        # Set input data (concat input and mask)
        prev_input = tf.concat(values=[prev_hid_data, x_mask],
                               axis=-1)

        # Set skim lstm
        with tf.variable_scope('lstm_{}'.format(l)) as vs:
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

        # Save hidden
        cur_fwd_hid, cur_bwd_hid = tf.split(value=cur_hid_data,
                                            num_or_size_splits=2,
                                            axis=2)
        fwd_hid_list.append(cur_fwd_hid)
        bwd_hid_list.append(cur_bwd_hid)

        # save action mask
        cur_fwd_act_mask, cur_bwd_act_mask = tf.split(value=cur_act_mask,
                                                      num_or_size_splits=2,
                                                      axis=2)
        fwd_act_mask_list.append(cur_fwd_act_mask)
        bwd_act_mask_list.append(cur_bwd_act_mask)

        # save action log prob
        cur_fwd_act_lgp, cur_bwd_act_lgp = tf.split(value=curr_act_lgp,
                                                    num_or_size_splits=2,
                                                    axis=2)
        fwd_act_lgp_list.append(cur_fwd_act_lgp)
        bwd_act_lgp_list.append(cur_bwd_act_lgp)

        # Set next input
        prev_hid_data = cur_hid_data

    # Set output layer
    with tf.variable_scope('output') as vs:
        output_linear = LinearCell(FLAGS.n_class)

    # Get sequence length and batch size
    seq_len = tf.shape(prev_hid_data)[0]
    num_samples = tf.shape(prev_hid_data)[1]
    output_feat_size = tf.shape(prev_hid_data)[2]

    # Get output logit (linear projection)
    output_logit = output_linear(tf.reshape(prev_hid_data, [-1, FLAGS.n_hidden]))
    output_logit = tf.reshape(output_logit, (seq_len, num_samples, FLAGS.n_class))

    # Get one-hot label
    y_1hot = tf.one_hot(y_data, depth=FLAGS.n_class)

    # Get parameters
    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    rl_params = []
    ml_params = []
    for var in model_params:
        if 'action' in var.name:
            rl_params.append(var)
        else:
            ml_params.append(var)

    # Define loss based cross entropy
    # Frame level
    ml_frame_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y_1hot, [-1, FLAGS.n_class]),
                                                            logits=tf.reshape(output_logit, [-1, FLAGS.n_class]))

    # Sample level
    ml_sample_loss = tf.reshape(ml_frame_loss, (seq_len, num_samples))
    ml_sample_loss = tf.reduce_sum(ml_sample_loss*tf.squeeze(x_mask, axis=-1), axis=0)/tf.reduce_sum(x_mask, axis=[0, 1])

    # Mean level
    ml_mean_loss = tf.reduce_sum(ml_sample_loss)/tf.to_float(num_samples)

    # Define frame-wise accuracy
    # Sample level
    sample_frame_accr = tf.to_float(tf.equal(tf.argmax(output_logit, axis=-1), tf.argmax(y_1hot, axis=-1)))
    sample_frame_accr = tf.reduce_sum(sample_frame_accr*tf.squeeze(x_mask, axis=-1))/tf.reduce_sum(x_mask, axis=[0, 1])

    # Mean level
    mean_frame_accr = tf.reduce_sum(sample_frame_accr)/tf.to_float(num_samples)

    # Define RL cost
    sample_reward = 1.0 - sample_frame_accr
    total_policy_cost = []
    total_baseline_cost = []
    # for each layer
    for i, act_data_list in enumerate(zip(fwd_hid_list,
                                          fwd_act_lgp_list,
                                          fwd_act_mask_list,
                                          bwd_hid_list,
                                          bwd_act_lgp_list,
                                          bwd_act_mask_list)):
        fwd_hid, fwd_lgp, fwd_mask, bwd_hid, bwd_lgp, bwd_mask = act_data_list

        # Forward pass
        # Get action mask and corresponding hidden state
        with tf.variable_scope('fwd_baseline_{}'.format(i)) as vs:
            fwd_W = tf.get_variable('W', [FLAGS.n_hidden, 1], dtype=fwd_hid.dtype)
            fwd_b = tf.get_variable('b', [FLAGS.n_hidden, 1], dtype=fwd_hid.dtype)
            tf.add_to_collection('weights', fwd_W)
            tf.add_to_collection('vars', fwd_W)
            tf.add_to_collection('vars', fwd_b)

        # set baseline based on the hidden (state)
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
        total_policy_cost.append([rl_fwd_policy_cost, [var for var in rl_params if str(i) in var.name and 'fwd' in var.name]])

        # Backward pass
        # Get action mask and corresponding hidden state
        with tf.variable_scope('bwd_baseline_{}'.format(i)) as vs:
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
        total_policy_cost.append([rl_bwd_policy_cost, [var for var in rl_params if str(i) in var.name and 'bwd' in var.name]])

    ml_cost = [ml_mean_loss, ml_params]

    return Graph(x_data=x_data,
                 x_mask=x_mask,
                 y_data=y_data,
                 init_state=init_state,
                 init_cntr=init_cntr,
                 mean_accr=mean_frame_accr,
                 mean_ml_cost=ml_mean_loss,
                 mean_rl_cost=tf.add_n([cost for cost, _ in total_policy_cost]),
                 mean_bl_cost=tf.add_n([cost for cost, _ in total_baseline_cost]),
                 ml_cost_param=ml_cost,
                 rl_cost_param_list=total_policy_cost,
                 bl_cost_param_list=total_baseline_cost)


def updater(model_graph,
            ml_updater,
            rl_updater,
            x_data,
            x_mask,
            y_data,
            session):
    outputs = session.run([model_graph.mean_accr,
                           model_graph.mean_ml_cost,
                           model_graph.mean_rl_cost,
                           model_graph.mean_bl_cost,
                           ml_updater,
                           rl_updater],
                          feed_dict={model_graph.x_data: x_data,
                                     model_graph.x_mask: x_mask,
                                     model_graph.y_data: y_data,
                                     model_graph.init_state: np.zeros(shape=(x_data.shape[1], FLAGS.n_hidden), dtype=x_data.dtype),
                                     model_graph.init_cntr: np.zeros(shape=(x_data.shape[1], 1), dtype=x_data.dtype)})

    mean_accr, ml_cost, rl_cost, bl_cost, _, _ = outputs
    return mean_accr, ml_cost, rl_cost, bl_cost


def evaluation(model_graph,
               session,
               dataset):
    # for each batch
    total_accr = 0.
    total_loss = 0.
    total_updates = 0.
    for b_idx, batch_data in dataset.get_epoch_iterator():
        # Get data x, y
        x_data, x_mask, _, _, y_data, _ = batch_data

        # Roll axis
        x_data = x_data.transpose((1, 0, 2))
        x_mask = x_mask.transpose((1, 0)).expand_dims(-1)
        y_data = y_data.transpose((1, 0))

        # Run model
        outputs = session.run([model_graph.mean_accr,
                               model_graph.mean_ml_cost],
                              feed_dict={model_graph.x_data: x_data,
                                         model_graph.x_mask: x_mask,
                                         model_graph.y_data: y_data,
                                         model_graph.init_state: np.zeros(shape=(x_data.shape[1], FLAGS.n_hidden),
                                                                          dtype=x_data.dtype),
                                         model_graph.init_cntr: np.zeros(shape=(x_data.shape[1], 1),
                                                                         dtype=x_data.dtype)})

        mean_accr, mean_ml_cost = outputs
        total_accr += mean_accr
        total_loss += mean_ml_cost
        total_updates += 1.0
    total_accr /= total_updates
    total_loss /= total_updates

    return total_accr, total_loss


def train_model():
    # Fix random seeds
    rand_seed = FLAGS.base_seed + FLAGS.add_seed
    tf.set_random_seed(rand_seed)
    np.random.seed(rand_seed)

    # Get module graph
    model_graph = build_graph(FLAGS)

    # Get model train max likelihood cost
    tf.add_to_collection('ml_cost', model_graph.ml_cost_param[0])
    ml_param = model_graph.ml_cost_param[1]

    # Get model train policy cost
    rl_param = []
    for c, v in model_graph.rl_cost_param_list + model_graph.bl_cost_param_list:
        tf.add_to_collection('rl_cost', c)
        rl_param.append(v)

    # Set weight decay
    if FLAGS.weight_decay > 0.0:
        weights_norm = tf.add_n([0.5*tf.nn.l2_loss(W) for W in ml_param if 'W' in W.name])
        weights_norm *= FLAGS.weight_decay
        tf.add_to_collection('ml_cost', weights_norm)

    # Set model max likelihood cost
    model_ml_cost = tf.add_n(tf.get_collection('ml_cost'), name='ml_cost')

    # Set model reinforce cost
    model_rl_cost = tf.add_n(tf.get_collection('rl_cost'), name='rl_cost')

    # Define global training step
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Set ml optimizer (Adam optimizer, in the original paper, we use 0.99 for beta2
    ml_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.ml_learning_rate,
                                    beta1=0.9,
                                    beta2=0.99,
                                    name='ml_optimizer')
    ml_grad = tf.gradients(ys=model_ml_cost, xs=ml_param, aggregation_method=2)

    # Set rl optimizer (SGD optimizer)
    rl_opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.rl_learning_rate,
                                               name='rl_optimizer')
    rl_grad = tf.gradients(ys=model_rl_cost, xs=rl_param, aggregation_method=2)

    # Set gradient clipping
    if FLAGS.ml_grad_clip > 0.0:
        ml_grad, _ = tf.clip_by_global_norm(t_list=ml_grad,
                                            clip_norm=FLAGS.ml_grad_clip)
    if FLAGS.ml_grad_clip > 0.0:
        rl_grad, _ = tf.clip_by_global_norm(t_list=rl_grad,
                                            clip_norm=FLAGS.rl_grad_clip)

    ml_update = ml_opt.apply_gradients(grads_and_vars=zip(ml_grad, ml_param),
                                       global_step=global_step)

    rl_update = rl_opt.apply_gradients(grads_and_vars=zip(rl_grad, rl_param),
                                       global_step=global_step)

    # Set dataset (sync_data(FLAGS))
    datasets = [FLAGS.train_dataset,
                FLAGS.valid_dataset,
                FLAGS.test_dataset]
    train_set, valid_set, test_set = [create_ivector_datastream(path=FLAGS.data_path,
                                                                which_set=dataset,
                                                                batch_size=FLAGS.batch_size)
                                      for dataset in datasets]

    # Set variable initializer
    init_op = tf.global_variables_initializer()

    # Set last model saver
    last_save_op = tf.train.Saver(max_to_keep=5)

    # Set best model saver
    best_save_op = tf.train.Saver(max_to_keep=5)

    # Get hardware config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Set session
    with tf.Session(config=config) as sess:
        # Init model
        sess.run(init_op)

        # Load checkpoint
        if FLAGS.start_from_ckpt:
            last_save_op = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, 'last_model.ckpt.meta'))
            last_save_op.restore(sess, os.path.join(FLAGS.log_dir, 'last_model.ckpt'))
            print("Restore from the last checkpoint. Restarting from %d step." % global_step.eval())

        # For each epoch
        accr_history = []
        ml_cost_history = []
        rl_cost_history = []
        bl_cost_history = []
        sum_cost_history = []

        best_accr = np.iinfo(np.int32).min
        for e_idx in xrange(FLAGS.n_epoch):
            # for each batch (update)
            for b_idx, batch_data in enumerate(train_set.get_epoch_iterator()):
                # Get data x, y
                x_data, x_mask, _, _, y_data, _ = batch_data

                # Roll axis
                x_data = x_data.transpose((1, 0, 2))
                x_mask = x_mask.transpose((1, 0)).expand_dims(-1)
                y_data = y_data.transpose((1, 0))

                # Get input size
                seq_len, n_sample, _ = x_data.shape

                # Update model
                mean_accr, ml_cost, rl_cost, bl_cost = updater(model_graph=model_graph,
                                                               ml_updater=ml_update,
                                                               rl_updater=rl_update,
                                                               x_data=x_data,
                                                               x_mask=x_mask,
                                                               y_data=y_data,
                                                               session=sess)
                accr_history.append(mean_accr)
                ml_cost_history.append(ml_cost)
                rl_cost_history.append(rl_cost)
                bl_cost_history.append(bl_cost)
                sum_cost_history.append(ml_cost + rl_cost + bl_cost)

                # Display results
                if global_step.eval() % FLAGS.display_freq == 0:
                    mean_accr = np.array(accr_history).mean()
                    mean_ml_cost = np.array(ml_cost_history).mean()
                    mean_rl_cost = np.array(rl_cost_history).mean()
                    mean_bl_cost = np.array(bl_cost_history).mean()
                    mean_sum_cost = np.array(sum_cost_history).mean()
                    print("====================================================")
                    print("Epoch " + str(e_idx) + ", Total Iter " + str(global_step.eval()))
                    print("----------------------------------------------------")
                    print("Average FER: {:.2f}%".format(mean_accr * 100))
                    print("Average  ML: {:.6f}".format(mean_ml_cost * 100))
                    print("Average  RL: {:.6f}".format(mean_rl_cost * 100))
                    print("Average  BL: {:.6f}".format(mean_bl_cost * 100))
                    print("Average SUM: {:.6f}".format(mean_sum_cost * 100))
                    last_ckpt = last_save_op.save(sess,
                                                  os.path.join(FLAGS.log_dir, "last_model.ckpt"),
                                                  global_step=global_step)
                    print("Last checkpointed in: %s" % last_ckpt)
                    accr_history = []
                    ml_cost_history = []
                    rl_cost_history = []
                    bl_cost_history = []
                    sum_cost_history = []

                # Evaluate model
                if global_step.eval() % FLAGS.evaluation_freq == 0:
                    # Monitor validation loss, accr
                    valid_accr, valid_cce = evaluation(model_graph=model_graph,
                                                       session=sess,
                                                       dataset=valid_set)
                    print("----------------------------------------------------")
                    print("Validation evaluation")
                    print("----------------------------------------------------")
                    print("FER: {:.2f}%".format((1.0-valid_accr)*100.))
                    print("CCE: {:.6f}".format(valid_cce))
                    print("----------------------------------------------------")
                    print("Best FER: {:.2f}%".format((1.0 - best_accr) * 100.))

                    # Save model
                    if best_accr < valid_accr:
                        best_accr = valid_accr
                        best_ckpt = best_save_op.save(sess,
                                                      os.path.join(FLAGS.log_dir, "best_model.ckpt"),
                                                      global_step=global_step)
                        print("Best checkpoint stored in: %s" % best_ckpt)
        print("Optimization Finished.")


def main(_):
    if not FLAGS.start_from_ckpt:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
    train_model()

if __name__ == '__main__':
    tf.app.run()
    main(_)

