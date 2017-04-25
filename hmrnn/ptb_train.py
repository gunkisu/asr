'''Train loop implementation using TensorFlow library.
'''
import os
import numpy as np
import sys
sys.path.insert(0, '..')
import theano
import theano.tensor as tensor
import time

from collections import OrderedDict
from data.ptb import TextIterator
from itertools import izip
from mixer import feed_dict
from mixer import gen_mask
from mixer import insert_item2dict
from mixer import init_tparams
from mixer import init_tparams_with_restored_value
from mixer import merge_dict
from mixer import reset_state
from mixer import save_npz
from mixer import save_npz2
from mixer import unzip


class TrainModel(object):
  """Class for training models on PTB
  character-level language modelling.
  """
  def __init__(self, FLAGS, build_graph=None, monitor=None, file_name='model'):
    """Some functions should be given by specific models"""
    if build_graph is None:
      raise NotImplementedError("Does not exist, provide the function.")
    if monitor is None:
      raise NotImplementedError("Does not exist, provide the function.")
    self._monitor = monitor
    self._build_graph = build_graph
    self._file_name = file_name
    self.FLAGS = FLAGS
    print("initial_learning_rate: %f" % self.FLAGS.learning_rate)
    print("batch_size: %d" % self.FLAGS.batch_size)

  def __call__(self):
    # Fix random seeds
    _seed = self.FLAGS.base_seed + self.FLAGS.add_seed
    np.random.seed(seed=_seed)
    # Prefixed names for save files
    prefix_name = os.path.join(self.FLAGS.log_dir, self._file_name)
    file_name = '%s.npz' % prefix_name
    best_file_name = '%s.best.npz' % prefix_name
    opt_file_name = '%s.grads.npz' % prefix_name
    best_opt_file_name = '%s.best.grads.npz' % prefix_name
    if self.FLAGS.start_from_ckpt and os.path.exists(file_name):
      self._ckpt_file_name = file_name
    # Declare summary
    summary = OrderedDict()
    # Initialize the variables
    f_prop, f_update, f_log_prob, f_debug, tparams, opt_tparams, \
        states, st_slope = self._build_graph(self.FLAGS)
    # Restore from checkpoint if FLAGS.start_from_ckpt is on
    if self.FLAGS.start_from_ckpt and os.path.exists(file_name):
      tparams = init_tparams_with_restored_value(tparams, file_name)
      model = np.load(file_name)
      for k, v in model.items():
        if 'summary' in k:
          summary[k] = list(v)
        if 'time' in k:
          summary[k] = list(v)
      global_step = model['global_step']
      epoch_step = model['epoch_step']
      batch_step = model['batch_step']
      print("Restore from the last checkpoint. "
            "Restarting from %d step." % global_step)
    else:
      global_step = 0
      epoch_step = 0
      batch_step = 0
    # Construct dataset objects
    train_set = TextIterator(which_set='train',
                             max_seq_len=self.FLAGS.max_seq_len,
                             batch_size=self.FLAGS.batch_size,
                             shuffle_every_epoch=1)
    if self.FLAGS.eval_train:
      train_infer_set = TextIterator(which_set='train',
                                     max_seq_len=self.FLAGS.max_seq_len,
                                     batch_size=self.FLAGS.batch_size,
                                     shuffle_every_epoch=0)
    else:
      train_infer_set = None
    valid_set = TextIterator(which_set='valid',
                             max_seq_len=self.FLAGS.max_seq_len,
                             batch_size=self.FLAGS.batch_size,
                             shuffle_every_epoch=0)
    if self.FLAGS.start_from_ckpt:
      _summary = self._monitor(f_log_prob, self.FLAGS, valid_set, None, states)
      _val_bits= _summary['val_bits']
      if _val_bits != summary['val_bits'][-1]:
        raise ValueError("Sanity check failed, check values do not match.")
      try:
        for cc in xrange(batch_step + 1):
          train_set.next()
      except:
        batch_step = 0
    best_params = None
    tr_costs = []
    _best_score = np.iinfo(np.int32).max
    # Keep training until max iteration
    print("Starting the optimization")
    for _epoch in xrange(self.FLAGS.n_epoch):
      reset_state(states)
      _n_exp = 0
      _time = time.time()
      __time = time.time()
      if self.FLAGS.start_from_ckpt and batch_step is not 0:
        pass
      else:
        batch_step = 0
      if self.FLAGS.use_slope_anneal:
        if _epoch <= self.FLAGS.n_anneal_epoch:
          new_slope = float(1. + (self.FLAGS.n_slope - 1) /
                            float(self.FLAGS.n_anneal_epoch) * _epoch)
          st_slope.set_value(new_slope)
          print("Changed the ST slope to : %f" % st_slope.get_value())
      for x in train_set:
        x, x_mask = gen_mask(x, max_seq_len=self.FLAGS.max_seq_len)
        _n_exp += self.FLAGS.batch_size
        # Run f-prop and optimization functions (backprop)
        cost = f_prop(x, x_mask)
        f_update(self.FLAGS.learning_rate)
        tr_costs.append(cost)
        if np.mod(global_step, self.FLAGS.display_freq) == 0:
          _time_spent = time.time() - _time
          tr_cost = np.array(tr_costs).mean()
          print("Epoch " + str(_epoch) + \
                ", Iter " + str(global_step) + \
                ", Average batch loss= " + "{:.6f}".format(tr_cost) + \
                ", Elapsed time= " + "{:.5f}".format(_time_spent))
          _time = time.time()
          tr_costs = []
        batch_step += 1
        global_step += 1
      # Monitor training/validation nats and bits
      _summary = self._monitor(f_log_prob, self.FLAGS, valid_set,
                               train_infer_set, states)
      feed_dict(summary, _summary)
      print("Train average nats= " + "{:.6f}".format(_summary['tr_nats']) + \
            ", Train average bits= " + "{:.6f}".format(_summary['tr_bits']) + \
            ", Valid average nats= " + "{:.6f}".format(_summary['val_nats']) + \
            ", Valid average bits= " + "{:.6f}".format(_summary['val_bits']) + \
            ", Elapsed time= " + "{:.5f}".format(time.time() - __time)) + \
            ", Observed examples= " + "{:d}".format(_n_exp)
      insert_item2dict(summary, 'time', _time_spent)
      # Save model
      _val_bits = summary['val_bits'][-1]
      if _val_bits < _best_score:
        _best_score = _val_bits
        # Save the best model
        best_params = unzip(tparams)
        if self.FLAGS.use_slope_anneal:
          best_params['st_slope'] = st_slope.get_value()
        save_npz(best_file_name, global_step, epoch_step, batch_step,
                 best_params, summary)
        # Save the gradients of best model
        best_opt_params = unzip(opt_tparams)
        save_npz2(best_opt_file_name, best_opt_params)
        print("Best checkpoint stored in: %s" % best_file_name)
      # Save the latest model
      params = unzip(tparams)
      if self.FLAGS.use_slope_anneal:
        params['st_slope'] = st_slope.get_value()
      save_npz(file_name, global_step, epoch_step, batch_step, params, summary)
      # Save the gradients of latest model
      opt_params = unzip(opt_tparams)
      save_npz2(opt_file_name, opt_params)
      print("Checkpointed in: %s" % file_name)
    print("Optimization Finished.")
