import argparse
import socket
import subprocess
import os, errno
import glob
import re
from collections import namedtuple
import random

import tensorflow as tf

import numpy as np
from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data

def add_datapath_args(parser):
    parser.add_argument('--no-copy', action='store_true', help='Do not copy the dataset to a local disk')
    parser.add_argument('--min-after-cache', default=1024, type=int, help='Size of mini-batch')
    parser.add_argument('--no-length-sort', action='store_true', help='Do not sort the dataset by sequence lengths')
    parser.add_argument('--data-path', default='/u/songinch/song/data/speech/wsj_fbank123.h5', help='Location of the dataset')
    parser.add_argument('--tmpdir', default='/Tmp/songinch/data/speech', help='Local temporary directory to store the dataset')

# Options shared by both training and testing
def add_general_args(parser):
    parser.add_argument('--n-batch', default=16, type=int, help='Size of mini-batch')
    parser.add_argument('--device', default='gpu', help='Simply set either `cpu` or `gpu`')

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_general_args(parser)
    add_datapath_args(parser)
    
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--lr2', default=0.001, type=float, help='Initial learning rate for RL')
    parser.add_argument('--factor', default=0.5, type=float, help='Learning rate scheduling factor')
    parser.add_argument('--n-epoch', default=15, type=int, help='Maximum number of epochs')
    parser.add_argument('--display-freq', default=50, type=int, help='Display frequency')
    parser.add_argument('--n-input', default=123, type=int, help='Number of RNN hidden units')
    parser.add_argument('--n-layer', default=1, type=int, help='Number of RNN hidden layers')
    parser.add_argument('--n-hidden', default=512, type=int, help='Number of RNN hidden units')
    parser.add_argument('--n-proj', default=0, type=int, help='Number of hidden units in projection layers')
    parser.add_argument('--n-class', default=3436, type=int, help='Number of target symbols')
    parser.add_argument('--n-action', default=5, type=int, help='Number of actions (max skim size)')
    parser.add_argument('--n-fast-action', default=0, type=int, help='Number of steps to skip in the fast action mode')
    parser.add_argument('--base-seed', default=20170309, type=int, help='Base random seed') 
    parser.add_argument('--add-seed', default=0, type=int, help='Add this amount to the base random seed')
    parser.add_argument('--logdir', default='skiprnn_test', help='Directory path to files')
    parser.add_argument('--train-dataset', default='train_si284', help='Training dataset')
    parser.add_argument('--valid-dataset', default='test_dev93', help='Validation dataset')
    parser.add_argument('--test-dataset', default='test_eval92', help='Test dataset')
    parser.add_argument('--discount-gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--use-layer-norm', action='store_true', help='Apply layer normalization')
    parser.add_argument('--use-sparse-reward', action='store_true', help='Use sparse rewards')
    parser.add_argument('--use-unimodal', action='store_true', help='Use unimodal output distribution')
    parser.add_argument('--tau', default=1.0, type=float, help='Temperature for softmax in unimodal distributions')
    parser.add_argument('--distribution', default='poisson', choices=['poisson', 'binomial'], help='Unimodal distribution to use')
    parser.add_argument('--no-stop-gradient', action='store_true', help='Do not stop gradient from flowing')
    parser.add_argument('--alpha', default=1.0, type=float, help='Coefficient for long skips')
    parser.add_argument('--beta', default=1.0, type=float, help='Hyperparameter for entropy regularizer')
    parser.add_argument('--max-to-keep', default=10, type=int, help='Number of models to keep')
    parser.add_argument('--use-label-miss-penalty', action='store_true', help='Penalize skips that miss label predictions')
    parser.add_argument('--label-miss-penalty', default=1, type=int, help='How much to penalize')
      
    return parser

def get_forward_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_general_args(parser)
    add_datapath_args(parser)    

    parser.add_argument('--dataset', default='test_dev93', help='Dataset to test')
    parser.add_argument('--wxfilename', default='ark:-', help='File to write')
    parser.add_argument('--metafile', default='best_model.ckpt-1000.meta', help='Model file to load')
    parser.add_argument('--show-progress', action='store_true', help='Whether to show progress')
    parser.add_argument('--sampling', action='store_true', help='Sample actions')

    return parser

def prepare_dir(args):
    if tf.gfile.Exists(args.logdir):
        tf.gfile.DeleteRecursively(args.logdir)
    tf.gfile.MakeDirs(args.logdir)

def get_gpuname():
    p = subprocess.Popen("nvidia-smi -q | grep 'Product Name'", shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    out = p.stdout.read()
#    import ipdb; ipdb.set_trace()
    gpuname = out.split(':')[1].strip()
    return gpuname

def print_host_info():
    print('Hostname: {}'.format(socket.gethostname()))
    print('GPU: {}'.format(get_gpuname()))

def prepare_dataset(args):
    sync_data(args)
    datasets = [args.train_dataset, args.valid_dataset, args.test_dataset]
    return [create_ivector_datastream(path=args.data_path, which_set=dataset, 
            batch_size=args.n_batch, min_after_cache=args.min_after_cache, length_sort=not args.no_length_sort) for dataset in datasets]

class Accumulator:
    def __init__(self):
        self.reset()
    
    def add(self, v, c):
        self.sum += v
        self.count += c

        self.last_sum = v
        self.last_count = c

    def avg(self):
        return self.sum / self.count
    
    def last_avg(self):
        return self.last_sum / self.last_count

    def reset(self):
        self.sum = 0.
        self.count = 0

        self.last_sum = 0.
        self.last_count = 0

    def __repr__(self):
        return 'Accumulator(sum={}, count={}, last_sum={}, last_count={})'.format(self.sum, self.count, self.last_sum, self.last_count)

def get_summary(summary_kinds):
    SummaryEntry = namedtuple('SummaryEntry', 'ph s') # placeholder and summary
    Summary = namedtuple('Summary', summary_kinds)

    tmp = []
    for sk in summary_kinds:
        ph = tf.placeholder(tf.float32)
        if 'image' in sk:
            tmp.append(SummaryEntry(ph, tf.summary.image(sk, ph)))
        else:
            tmp.append(SummaryEntry(ph, tf.summary.scalar(sk, ph)))
    summary = Summary._make(tmp)
    
    return summary


def hardlink_force(target, link_name):
    try:
        os.link(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.link(target, link_name)
        else:
            raise e

def init_savers(args):
    save_op = tf.train.Saver(max_to_keep=args.n_epoch)
    best_save_op = tf.train.Saver(max_to_keep=args.n_epoch)
    return save_op, best_save_op

def link_to_best_model(best_ckpt, args):
    hardlink_force(os.path.abspath('{}.meta'.format(best_ckpt)), os.path.join(args.logdir, 'best_model.meta'))

def get_model_size(trainable_variables):
    return float(np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables])) / 1000000

def delayed(output, seq_y_data, seq_x_mask, delay):
    return output[:,delay:,:], seq_y_data[:,delay:], seq_x_mask[:,delay:]


def find_model(metafile):
    if os.path.isdir(metafile):
        model_list = glob.glob(os.path.join(metafile, 'best_model*meta*'))
        pat = re.compile('-(\d+).meta')

        tmp_model_list = []
        for model in model_list:
            m = pat.search(model)
            if m:
                tmp_model_list.append((int(m.group(1)), model))

        tmp_model_list.sort(reverse=True)
        iter_no, model = tmp_model_list[0]
        return model
    else:
        return metafile

def reduce_lr(lr, factor, sess):
    if factor < 1.0:
        assign_op = tf.assign(lr, lr * factor)
        sess.run(assign_op)

def win_iter(batch, win_size, right_context=0):
    n_seq = batch[0].shape[1]

    if win_size == 0:
        yield batch
    else:
        for i in range(0, n_seq, win_size):
            from_idx = i
            to_idx = i+win_size+right_context
            
            yield (src[:,from_idx:to_idx] for src in batch)
   

def skip_frames_fixed(batch, every_n, return_first=False, return_start_idx=False):
    assert every_n > 0

    if return_first:
        start_idx = 0
    else:
        start_idx = random.randint(0, every_n-1)

    new_batch = []

    for src_data in batch:
        new_src_data = []
        for ex in src_data:
            new_src_data.append(ex[start_idx::every_n])

        new_batch.append(np.asarray(new_src_data))

    if return_start_idx:
        return new_batch, start_idx
    else:
        return new_batch

class LinearVF(object):
    def __init__(self, num_iter=1):
        self.coeffs = None
        self.reg_coeff = 0.0
        self.num_iter = num_iter

    def _features(self, x):
        o = x.astype('float32')
#        return np.concatenate([o, o**2, o**3])
        return np.concatenate([o])

    def get_featmat(self, X):
        return np.asarray([self._features(x) for x in X])

    def fit(self, X, returns):
        featmat = self.get_featmat(X)
        reg_coeff = self.reg_coeff
        for _ in range(self.num_iter):
            # Equation 3.28 in PRML
            self.coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self.coeffs)):
                break
            reg_coeff *= 2

    def predict(self, X):
        # [n_batch * n_seq, n_feat]
        
        if self.coeffs is None: 
            return np.zeros(X.shape[0]) # zeros of [n_batch * n_seq]

        return self.get_featmat(X).dot(self.coeffs)