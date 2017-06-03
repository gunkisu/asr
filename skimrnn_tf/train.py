import os
import numpy as np
import tensorflow as tf
from libs.utils import StopWatch
from data.fuel_utils import create_ivector_datastream
from model import LinearCell, SkimLSTMModule
from collections import namedtuple

tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model size
flags.DEFINE_integer('n_input', 123, 'Number of inputs')
flags.DEFINE_integer('n_hidden', 128, 'Number of LSTM hidden units')
flags.DEFINE_integer('n_class', 3436, 'Number of target symbols')
flags.DEFINE_integer('n_layer', 3, 'Number of layers')
flags.DEFINE_float('forget_bias', 1.0, 'forget bias')

flags.DEFINE_boolean('use_skim', True, 'use skim')

flags.DEFINE_boolean('use_input', False, 'set input as state')

# Action size
flags.DEFINE_integer('n_read', 1, 'Number of minimum read')
flags.DEFINE_integer('n_action', 4, 'Number of maximum skim')

# Random Seet
flags.DEFINE_integer('base_seed', 2222, 'Base random seed')
flags.DEFINE_integer('add_seed', 0, 'Add this amount to the base random seed')

# Learning
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay factor')
flags.DEFINE_float('grad_clip', 1.0, 'Gradient norm clipping')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('batch_size', 16, 'Size of mini-batch')
flags.DEFINE_integer('n_epoch', 200, 'Maximum number of epochs')
flags.DEFINE_integer('max_length', 100, 'Maximum number of epochs')

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
                   'mean_loss',
                   'ml_cost',
                   'rl_cost',
                   'bl_cost',
                   'read_ratio_list']

Graph = namedtuple('Graph', ' '.join(graph_attr_list))


def build_graph(FLAGS):
    # Define input data
    with tf.device(FLAGS.device):
        # input sequence (seq_len, num_samples, num_input)
        x_data = tf.placeholder(dtype=tf.float32,
                                shape=(None, None, FLAGS.n_input),
                                name='x_data')

        # input mask (seq_len, num_samples)
        x_mask = tf.placeholder(dtype=tf.float32,
                                shape=(None, None),
                                name='x_mask')

        # gt label (seq_len, num_samples)
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

    # Get one-hot label
    y_1hot = tf.one_hot(y_data, depth=FLAGS.n_class)

    # Get sequence length and batch size
    seq_len = tf.shape(x_data)[0]
    num_samples = tf.shape(x_data)[1]

    # For each layer
    policy_data_list = []
    prev_hid_data = x_data
    for l in range(FLAGS.n_layer):
        # Set input data (concat input and mask)
        prev_input = tf.concat(values=[prev_hid_data, tf.expand_dims(x_mask, axis=-1)],
                               axis=-1)

        # Set skim lstm
        with tf.variable_scope('lstm_{}'.format(l)) as vs:
            skim_lstm = SkimLSTMModule(num_units=FLAGS.n_hidden,
                                       max_skims=FLAGS.n_action,
                                       min_reads=FLAGS.n_read,
                                       forget_bias=FLAGS.forget_bias,
                                       use_input=FLAGS.use_input,
                                       use_skim=FLAGS.use_skim)

            # Run bidir skim lstm
            outputs = skim_lstm(inputs=prev_input,
                                init_state=[init_state, init_cntr],
                                use_bidir=True)

        # Get output
        hid_data, read_mask, act_mask, act_lgp = outputs

        # Split data
        fwd_hid_data, bwd_hid_data = tf.split(value=hid_data, num_or_size_splits=2, axis=2)
        fwd_read_mask, bwd_read_mask = tf.split(value=read_mask, num_or_size_splits=2, axis=2)
        fwd_act_mask, bwd_act_mask = tf.split(value=act_mask, num_or_size_splits=2, axis=2)
        fwd_act_lgp, bwd_act_lgp = tf.split(value=act_lgp, num_or_size_splits=2, axis=2)

        # Set summary
        tf.summary.image(name='fwd_results_{}'.format(l),
                         tensor=tf.concat(values=[tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(x_mask, [1, 0]), axis=-1), axis=1), [1, 20, 1, 1]),
                                                  tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(tf.to_float(y_data)/tf.to_float(FLAGS.n_class), [1, 0]), axis=-1), axis=1), [1, 20, 1, 1]),
                                                  tf.tile(tf.expand_dims(tf.transpose(fwd_read_mask, [1, 0, 2]), axis=1), [1, 20, 1, 1]),
                                                  tf.tile(tf.expand_dims(tf.transpose(fwd_act_mask, [1, 0, 2]), axis=1), [1, 20, 1, 1]),],
                                          axis=1))
        tf.summary.image(name='bwd_results_{}'.format(l),
                         tensor=tf.concat(values=[tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(x_mask, [1, 0]), axis=-1), axis=1), [1, 20, 1, 1]),
                                                  tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(tf.to_float(y_data)/tf.to_float(FLAGS.n_class), [1, 0]), axis=-1), axis=1), [1, 20, 1, 1]),
                                                  tf.tile(tf.expand_dims(tf.transpose(bwd_read_mask, [1, 0, 2]), axis=1), [1, 20, 1, 1]),
                                                  tf.tile(tf.expand_dims(tf.transpose(bwd_act_mask, [1, 0, 2]), axis=1), [1, 20, 1, 1]),],
                                          axis=1))

        # Set baseline
        with tf.variable_scope("fwd_baseline_{}".format(l)) as vs:
            fwd_policy_input = tf.reshape(tf.stop_gradient(fwd_hid_data), [-1, FLAGS.n_hidden])
            fwd_baseline_cell = LinearCell(num_units=1)
            fwd_basline = fwd_baseline_cell(fwd_policy_input)
            fwd_basline = tf.reshape(fwd_basline, [seq_len, num_samples])

        with tf.variable_scope("bwd_baseline_{}".format(l)) as vs:
            bwd_policy_input = tf.reshape(tf.stop_gradient(bwd_hid_data), [-1, FLAGS.n_hidden])
            bwd_baseline_cell = LinearCell(num_units=1)
            bwd_basline = bwd_baseline_cell(bwd_policy_input)
            bwd_basline = tf.reshape(bwd_basline, [seq_len, num_samples])

        # Set next input
        prev_hid_data = hid_data

        # Save data
        policy_data_list.append([tf.squeeze(fwd_read_mask), tf.squeeze(fwd_act_mask), fwd_act_lgp, fwd_basline])
        policy_data_list.append([tf.squeeze(bwd_read_mask), tf.squeeze(bwd_act_mask), bwd_act_lgp, bwd_basline])

    # Set output layer
    with tf.variable_scope('output') as vs:
        output_cell = LinearCell(FLAGS.n_class)
        output_logit = output_cell(tf.reshape(prev_hid_data, [-1, 2*FLAGS.n_hidden]))
        output_logit = tf.reshape(output_logit, (seq_len, num_samples, FLAGS.n_class))

    # Frame-wise cross entropy
    frame_cce = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y_1hot, [-1, FLAGS.n_class]),
                                                        logits=tf.reshape(output_logit, [-1, FLAGS.n_class]))
    frame_cce *= tf.reshape(x_mask, [-1, ])

    # Frame mean cce
    mean_frame_cce = tf.reduce_sum(frame_cce) / tf.reduce_sum(x_mask)
    tf.summary.scalar(name='frame_cce', tensor=mean_frame_cce)

    # Model cce
    model_cce = tf.reduce_sum(frame_cce) / tf.to_float(num_samples)

    # Frame-wise accuracy
    frame_accr = tf.to_float(tf.equal(tf.argmax(output_logit, axis=-1), tf.argmax(y_1hot, axis=-1))) * x_mask
    sample_frame_accr = tf.reduce_sum(frame_accr, axis=0)/tf.reduce_sum(x_mask, axis=0)
    mean_frame_accr = tf.reduce_sum(frame_accr)/tf.reduce_sum(x_mask)
    tf.summary.scalar(name='frame_accr', tensor=mean_frame_accr)

    # Sample-wise REWARD
    sample_reward = sample_frame_accr

    # Define policy cost for each network
    baseline_cost_list = []
    policy_cost_list = []
    read_ratio_list = []
    for i, policy_data in enumerate(policy_data_list):
        # Get data
        read_mask, act_mask, act_lgp, baseline = policy_data

        # Get read ratio
        read_ratio = tf.reduce_sum(read_mask, axis=0)/tf.reduce_sum(x_mask, axis=0)
        skim_ratio = 1.0 - read_ratio

        # combine reward (frame accuracy and skim ratio)
        original_reward = (sample_reward + skim_ratio*0.0)

        # revised with baseline
        revised_reward = (tf.expand_dims(original_reward, axis=0) - baseline)*act_mask

        # baseline cost
        baseline_cost = tf.reduce_sum(tf.square(revised_reward))

        # policy cost
        policy_cost = tf.stop_gradient(revised_reward)*tf.reduce_sum(act_lgp, axis=-1)*act_mask
        policy_cost = -tf.reduce_sum(policy_cost)/tf.to_float(num_samples)

        # Save values
        baseline_cost_list.append(baseline_cost)
        policy_cost_list.append(policy_cost)
        read_ratio_list.append(tf.reduce_mean(read_ratio, keep_dims=True))

        tf.summary.scalar(name='frame_read_ratio_{}'.format(i), tensor=tf.reduce_mean(read_ratio))

    tf.summary.scalar(name='policy_cost', tensor=tf.add_n(policy_cost_list))
    tf.summary.scalar(name='baseline_cost', tensor=tf.add_n(baseline_cost_list))

    return Graph(x_data=x_data,
                 x_mask=x_mask,
                 y_data=y_data,
                 init_state=init_state,
                 init_cntr=init_cntr,
                 mean_accr=mean_frame_accr,
                 mean_loss=mean_frame_cce,
                 ml_cost=model_cce,
                 rl_cost=tf.add_n(policy_cost_list),
                 bl_cost=tf.add_n(baseline_cost_list),
                 read_ratio_list=tf.concat(read_ratio_list, axis=0))


# Single update
def updater(model_graph,
            model_updater,
            x_data,
            x_mask,
            y_data,
            summary,
            session):

    # Run session
    outputs = session.run([model_graph.mean_accr, # frame-wise accuracy
                           model_graph.mean_loss, # frame-wise cost
                           model_graph.ml_cost, # full ml cost
                           model_graph.rl_cost, # full rl cost
                           model_graph.bl_cost, # full bl cost
                           model_graph.read_ratio_list, # read_ratio_list
                           model_updater, # model update
                           summary],
                          feed_dict={model_graph.x_data: x_data,
                                     model_graph.x_mask: x_mask,
                                     model_graph.y_data: y_data,
                                     model_graph.init_state: np.zeros(shape=(x_data.shape[1], FLAGS.n_hidden), dtype=x_data.dtype),
                                     model_graph.init_cntr: np.zeros(shape=(x_data.shape[1], 1), dtype=x_data.dtype)})

    mean_accr, mean_loss, ml_cost, rl_cost, bl_cost, read_ratio, _, summary_output = outputs
    return mean_accr, mean_loss, ml_cost, rl_cost, bl_cost, read_ratio, summary_output


# Model evaluation
def evaluation(model_graph,
               session,
               dataset):
    # for each batch
    total_accr = 0.
    total_loss = 0.
    total_read_ratio = 0.
    total_updates = 0.
    for b_idx, batch_data in enumerate(dataset.get_epoch_iterator()):
        # Get data x, y
        x_data, x_mask, _, _, y_data, _ = batch_data

        # Roll axis
        x_data = x_data.transpose((1, 0, 2))
        x_mask = x_mask.transpose((1, 0))
        y_data = y_data.transpose((1, 0))

        # Run model
        outputs = session.run([model_graph.mean_accr,
                               model_graph.mean_loss,
                               model_graph.read_ratio_list],
                              feed_dict={model_graph.x_data: x_data,
                                         model_graph.x_mask: x_mask,
                                         model_graph.y_data: y_data,
                                         model_graph.init_state: np.zeros(shape=(x_data.shape[1], FLAGS.n_hidden), dtype=x_data.dtype),
                                         model_graph.init_cntr: np.zeros(shape=(x_data.shape[1], 1), dtype=x_data.dtype)})

        mean_accr, mean_loss, read_ratio = outputs
        total_accr += mean_accr
        total_loss += mean_loss
        total_read_ratio += read_ratio.mean()
        total_updates += 1.0

    total_accr /= total_updates
    total_loss /= total_updates

    return total_accr, total_loss


# Train model
def train_model():
    sw = StopWatch()

    # Fix random seeds
    rand_seed = FLAGS.base_seed + FLAGS.add_seed
    tf.set_random_seed(rand_seed)
    np.random.seed(rand_seed)

    # Get module graph
    model_graph = build_graph(FLAGS)

    # Get model parameter
    model_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Set weight decay
    if FLAGS.weight_decay > 0.0:
        l2_cost = tf.add_n([0.5*tf.nn.l2_loss(W) for W in model_param
                                 if 'W' in W.name and 'action' not in W.name and 'baseline' not in W.name])
        l2_cost *= FLAGS.weight_decay
    else:
        l2_cost = 0.0

    # Set total cost
    model_total_cost = model_graph.ml_cost + model_graph.rl_cost + model_graph.bl_cost + l2_cost

    # Define global training step
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Set ml optimizer (Adam optimizer, in the original paper, we use 0.99 for beta2
    model_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                       name='model_optimizer')
    model_grad = tf.gradients(ys=model_total_cost, xs=model_param, aggregation_method=2)

    # Set gradient clipping
    if FLAGS.grad_clip > 0.0:
        model_grad, _ = tf.clip_by_global_norm(t_list=model_grad,
                                               clip_norm=FLAGS.grad_clip)

    model_update = model_opt.apply_gradients(grads_and_vars=zip(model_grad,
                                                                model_param),
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
        # Get summary
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             sess.graph)

        # Init model
        sess.run(init_op)

        # Load checkpoint
        if FLAGS.start_from_ckpt:
            last_save_op = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, 'last_model.ckpt.meta'))
            last_save_op.restore(sess, os.path.join(FLAGS.log_dir, 'last_model.ckpt'))
            print("Restore from the last checkpoint. Restarting from %d step." % global_step.eval())

        # For each epoch
        accr_history = []
        loss_history = []
        ml_cost_history = []
        rl_cost_history = []
        bl_cost_history = []
        sum_cost_history = []

        best_accr = 0.0
        sw.reset()
        for e_idx in xrange(FLAGS.n_epoch):
            # for each batch (update)
            for b_idx, batch_data in enumerate(train_set.get_epoch_iterator()):
                # Get data x, y
                x_data, x_mask, _, _, y_data, _ = batch_data

                # Roll axis
                x_data = x_data.transpose((1, 0, 2))
                x_mask = x_mask.transpose((1, 0))
                y_data = y_data.transpose((1, 0))

                # Update model
                mean_accr, mean_loss, ml_cost, rl_cost, bl_cost, read_ratio, summary_output \
                    = updater(model_graph=model_graph,
                              model_updater=model_update,
                              x_data=x_data,
                              x_mask=x_mask,
                              y_data=y_data,
                              summary=merged_summary,
                              session=sess)

                # write summary
                train_writer.add_summary(summary_output, global_step.eval())

                accr_history.append(mean_accr)
                loss_history.append(mean_loss)
                ml_cost_history.append(ml_cost)
                rl_cost_history.append(rl_cost)
                bl_cost_history.append(bl_cost)
                sum_cost_history.append(ml_cost + rl_cost + bl_cost)

                # Display results
                if global_step.eval() % FLAGS.display_freq == 0:
                    mean_accr = np.array(accr_history).mean()
                    mean_loss = np.array(loss_history).mean()
                    mean_ml_cost = np.array(ml_cost_history).mean()
                    mean_rl_cost = np.array(rl_cost_history).mean()
                    mean_bl_cost = np.array(bl_cost_history).mean()
                    mean_sum_cost = np.array(sum_cost_history).mean()
                    print("====================================================")
                    print("Epoch " + str(e_idx) + ", Total Iter " + str(global_step.eval()))
                    print("----------------------------------------------------")
                    print("Average FER: {:.2f}%".format((1.0-mean_accr) * 100))
                    print("Average CCE: {:.6f}".format(mean_loss))
                    print("Average  ML: {:.6f}".format(mean_ml_cost))
                    if FLAGS.use_skim:
                        print("Average  RL: {:.6f}".format(mean_rl_cost))
                        print("Average  BL: {:.6f}".format(mean_bl_cost))
                        print("Average SUM: {:.6f}".format(mean_sum_cost))
                    print("Read ratio: ", read_ratio)
                    sw.print_elapsed()
                    sw.reset()
                    last_ckpt = last_save_op.save(sess,
                                                  os.path.join(FLAGS.log_dir, "last_model.ckpt"),
                                                  global_step=global_step)
                    print("Last checkpointed in: %s" % last_ckpt)
                    accr_history = []
                    loss_history = []
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
                    # Save model
                    if best_accr < valid_accr:
                        best_accr = valid_accr
                        best_ckpt = best_save_op.save(sess,
                                                      os.path.join(FLAGS.log_dir, "best_model.ckpt"),
                                                      global_step=global_step)
                        print("Best checkpoint stored in: %s" % best_ckpt)

                    print("----------------------------------------------------")
                    print("Validation evaluation")
                    print("----------------------------------------------------")
                    print("FER: {:.2f}%".format((1.0-valid_accr)*100.))
                    print("CCE: {:.6f}".format(valid_cce))
                    print("----------------------------------------------------")
                    print("Best FER: {:.2f}%".format((1.0 - best_accr) * 100.))

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

