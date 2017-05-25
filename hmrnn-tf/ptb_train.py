'''Train loop implementation using TensorFlow library.
'''
import os
import numpy as np
import sys
sys.path.insert(0, '..')
import tensorflow as tf
import time

from collections import OrderedDict
from data.ptb import TextIterator
from mixer import gen_mask
from mixer import insert_item2dict
from mixer import save_npz2


class TrainModel(object):
  """Class for training models on PTB
  character-level language modelling.
  """
  def __init__(self, FLAGS, build_graph=None, monitor=None,
               initial_values=None, sess_wrapper=None,
               define_summary=None, add_monitor_op=None,
               file_name='model'):
    """Some functions should be given by specific models"""
    if build_graph is None:
      raise NotImplementedError("Does not exist, provide the function.")
    if monitor is None:
      raise NotImplementedError("Does not exist, provide the function.")
    if initial_values is None:
      raise NotImplementedError("Does not exist, provide the function.")
    if sess_wrapper is None:
      raise NotImplementedError("Does not exist, provide the function.")
    if define_summary is None:
      raise NotImplementedError("Does not exist, provide the function.")
    self._monitor = monitor
    self._build_graph = build_graph
    self._initial_states = initial_values
    self._sess_wrapper = sess_wrapper
    self._define_summary = define_summary
    self._add_monitor_op = add_monitor_op
    self._file_name = file_name
    self.FLAGS = FLAGS
    print("initial_learning_rate: %f" % self.FLAGS.learning_rate)
    print("batch_size: %d" % self.FLAGS.batch_size)

  def __call__(self):
    # Reuse variables if needed
    tf.get_variable_scope()._reuse = None
    # Fix random seeds
    _seed = self.FLAGS.base_seed + self.FLAGS.add_seed
    tf.set_random_seed(_seed)
    np.random.seed(_seed)
    # Prefixed names for save files
    prefix_name = os.path.join(self.FLAGS.log_dir, self._file_name)
    file_name = '%s.npz' % prefix_name
    # Declare summary
    summary = OrderedDict()
    # Training objective should always come at the front
    G = self._build_graph(self.FLAGS)
    _cost = tf.reduce_mean(G[0])
    tf.add_to_collection('losses', _cost)
    if self.FLAGS.weight_decay > 0.0:
      with tf.variable_scope('weight_norm') as scope:
        weights_norm = tf.reduce_sum(input_tensor=self.FLAGS.weight_decay *
          tf.stack([2*tf.nn.l2_loss(W) for W in tf.get_collection('weights')]),
          name='weights_norm'
          )
      tf.add_to_collection('losses', weights_norm)
    _cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
    # Define non-trainable variables to track the progress
    global_step = tf.Variable(0, trainable=False, name="global_step")
    # For Adam optimizer, in the original paper, we use 0.99 for beta2
    opt_func = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate,
                                      beta1=0.9, beta2=0.99)
    if not self.FLAGS.grad_clip:
      # Without clipping
      t_step = opt_func.minimize(_cost, global_step=global_step)
    else:
      # Apply gradient clipping using global norm
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(_cost, tvars,
                                                     aggregation_method=2),
                                        clip_norm=1.0)
      t_step = opt_func.apply_gradients(zip(grads, tvars),
                                        global_step=global_step)
    # Construct dataset objects
    train_set = TextIterator(which_set='train',
                             max_seq_len=self.FLAGS.max_seq_len,
                             batch_size=self.FLAGS.batch_size,
                             shuffle_every_epoch=True)
    if self.FLAGS.eval_train:
      train_infer_set = TextIterator(which_set='train',
                                     max_seq_len=self.FLAGS.max_seq_len,
                                     batch_size=self.FLAGS.batch_size)
    else:
      train_infer_set = None
    valid_set = TextIterator(which_set='valid',
                             max_seq_len=self.FLAGS.max_seq_len,
                             batch_size=self.FLAGS.batch_size,
                             shuffle_every_epoch=True)
    init_op = tf.global_variables_initializer()
    save_op = tf.train.Saver(max_to_keep=5)
    best_save_op = tf.train.Saver(max_to_keep=5)
    with tf.name_scope("per_step_eval"):
      # Add batch level nats to summaries
      step_summary = tf.summary.scalar("tr_nats", tf.reduce_mean(G[0]))
    # Add monitor ops if exist
    if self._add_monitor_op is not None:
        monitor_op = self._add_monitor_op(self.FLAGS)
    else:
        monitor_op = None
    S = self._define_summary()
    with tf.Session() as sess:
      # Initializing the variables
      sess.run(init_op)
      if self.FLAGS.start_from_ckpt:
        save_op = tf.train.import_meta_graph(os.path.join(self.FLAGS.log_dir,
                                                          'model.ckpt.meta'))
        save_op.restore(sess, os.path.join(self.FLAGS.log_dir, 'model.ckpt'))
        print("Restore from the last checkpoint. "
              "Restarting from %d step." % global_step.eval())
      # Declare summary writer
      summary_writer = tf.summary.FileWriter(self.FLAGS.log_dir,
                                             flush_secs=5.0)
      tr_costs = []
      _best_score = np.iinfo(np.int32).max
      # Keep training until max iteration
      for _epoch in xrange(self.FLAGS.n_epoch):
        _feed_states = self._initial_states(self.FLAGS)
        _n_exp = 0
        _time = time.time()
        __time = time.time()
        for x in train_set:
          x, x_mask = gen_mask(x, self.FLAGS.max_seq_len)
          _n_exp += self.FLAGS.batch_size
          # Run optimization op (backprop)
          _tr_cost, _step_summary, _feed_states = \
                  self._sess_wrapper(x, x_mask, t_step, step_summary, sess,
                                     G, _feed_states)
          # Write step level logs
          summary_writer.add_summary(_step_summary, global_step.eval())
          tr_costs.append(_tr_cost.mean())
          if global_step.eval() % self.FLAGS.display_freq == 0:
            tr_cost = np.array(tr_costs).mean()
            print("Epoch " + str(_epoch) + \
                  ", Iter " + str(global_step.eval()) + \
                  ", Average batch loss= " + "{:.6f}".format(tr_cost) + \
                  ", Elapsed time= " + "{:.5f}".format(time.time() - _time))
            _time = time.time()
            tr_costs = []
        # Monitor training/validation nats and bits
        _feed_states = self._initial_states(self.FLAGS)
        _tr_nats, _tr_bits, _val_nats, _val_bits = \
                self._monitor(G, sess, train_infer_set, valid_set,
                              _feed_states, self.FLAGS, summary,
                              S, summary_writer, global_step.eval(), monitor_op)
        _time_spent = time.time() - __time
        print("Train average nats= " + "{:.6f}".format(_tr_nats) + \
              ", Train average bits= " + "{:.6f}".format(_tr_bits) + \
              ", Valid average nats= " + "{:.6f}".format(_val_nats) + \
              ", Valid average bits= " + "{:.6f}".format(_val_bits) + \
              ", Elapsed time= " + "{:.5f}".format(_time_spent)) + \
              ", Observed examples= " + "{:d}".format(_n_exp)
        insert_item2dict(summary, 'time', _time_spent)
        save_npz2(file_name, summary)
        # Save model
        if _val_bits < _best_score:
          _best_score = _val_bits
          best_ckpt = best_save_op.save(sess, os.path.join(self.FLAGS.log_dir,
                                                           "best_model.ckpt"),
                                        global_step=global_step)
          print("Best checkpoint stored in: %s" % best_ckpt)
        ckpt = save_op.save(sess, os.path.join(self.FLAGS.log_dir,
                            "model.ckpt"),
                            global_step=global_step)
        print("Checkpointed in: %s" % ckpt)
        _epoch_summary = sess.run([S[0][0]],
                                  feed_dict={S[1][0]: _best_score})
        summary_writer.add_summary(_epoch_summary[0], global_step.eval())
      summary_writer.close()
      print("Optimization Finished.")
