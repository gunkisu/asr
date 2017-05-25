'''This script is a Penn Treebank data generator that yields
a sequence of integers.
'''
import numpy as np


class TextIterator:
  """Simple Bitext iterator.
  """
  def __init__(self, which_set, max_seq_len=100, batch_size=64,
               shuffle_every_epoch=False):
    self.batch_size = batch_size
    self.shuffle_every_epoch = shuffle_every_epoch
    self.which_set = which_set
    self.max_seq_len = max_seq_len
    path = '/u/songinch/song/data/ptb/char_penntree.npz'
    data = np.load(path)
    self.data = data[which_set]
    self.size = int(len(self.data) / self.batch_size) * self.batch_size
    self.data_ = self.chop()
    self.idx = 0


  def chop(self):
    # Reshape to non-overlapping examples
    if self.shuffle_every_epoch:
      roll_step = np.random.randint(len(self.data))
      data = np.roll(self.data, axis=0, shift=roll_step)
    else:
      data = self.data
    batch_data = data[:self.size].reshape(self.batch_size, -1).transpose()
    # [sequence_temporal_direction, batch_size]
    return batch_data


  def __iter__(self):
    return self


  def next(self):
    if self.idx * (self.max_seq_len - 1) >= self.data_.shape[0]:
      self.idx = 0
      self.data_ = self.chop()
      raise StopIteration
    else:
      # It is important to make one symbol to overlap.
      # If not, you are basically skipping a symbol.
      batch = self.data_[self.idx*(self.max_seq_len-1):self.idx*(self.max_seq_len-1)+self.max_seq_len]
      batch = batch.astype('int32')
      self.idx += 1
    return batch
