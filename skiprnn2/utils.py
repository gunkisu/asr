from __future__ import print_function

import argparse
import socket
import tensorflow as tf
import subprocess
import os, errno
from collections import namedtuple

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
    parser.add_argument('--n-batch', default=32, type=int, help='Size of mini-batch')
    parser.add_argument('--device', default='gpu', help='Simply set either `cpu` or `gpu`')

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_general_args(parser)
    add_datapath_args(parser)
    
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--lr2', default=0.001, type=float, help='Initial learning rate for RL')
    parser.add_argument('--n-epoch', default=20, type=int, help='Maximum number of epochs')
    parser.add_argument('--display-freq', default=50, type=int, help='Display frequency')
    parser.add_argument('--n-input', default=123, type=int, help='Number of RNN hidden units')
    parser.add_argument('--n-layer', default=1, type=int, help='Number of RNN hidden layers')
    parser.add_argument('--n-hidden', default=1024, type=int, help='Number of RNN hidden units')
    parser.add_argument('--n-proj', default=0, type=int, help='Number of hidden units in projection layers')
    parser.add_argument('--n-class', default=3436, type=int, help='Number of target symbols')
    parser.add_argument('--n-embedding', default=0, type=int, help='Embedding size')
    parser.add_argument('--n-action', default=6, type=int, help='Number of actions (max skim size)')
    parser.add_argument('--n-fast-action', default=0, type=int, help='Number of steps to skip in the fast action mode')
    parser.add_argument('--base-seed', default=20170309, type=int, help='Base random seed') 
    parser.add_argument('--add-seed', default=0, type=int, help='Add this amount to the base random seed')
    parser.add_argument('--logdir', default='skiprnn_test', help='Directory path to files')
    parser.add_argument('--train-dataset', default='train_si284', help='Training dataset')
    parser.add_argument('--valid-dataset', default='test_dev93', help='Validation dataset')
    parser.add_argument('--test-dataset', default='test_eval92', help='Test dataset')
    parser.add_argument('--discount-gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--use-layer-norm', action='store_true', help='Apply layer normalization')
    
    return parser

def get_forward_argparser():
    parser = argparse.ArgumentParser()

    add_general_args(parser)
    add_datapath_args(parser)    

    parser.add_argument('--dataset', default='test_dev93', help='Dataset to test')
    parser.add_argument('--wxfilename', default='ark:-', help='File to write')
    parser.add_argument('--metafile', default='best_model.ckpt-1000.meta', help='Model file to load')
    parser.add_argument('--show-progress', action='store_true', help='Whether to show progress')

    return parser

def prepare_dir(args):
    if tf.gfile.Exists(args.logdir):
        tf.gfile.DeleteRecursively(args.logdir)
    tf.gfile.MakeDirs(args.logdir)

def get_gpuname():
    p = subprocess.Popen("nvidia-smi -q | grep 'Product Name'", shell=True, stdout=subprocess.PIPE)
    out = p.stdout.read()
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


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def link_to_best_model(best_ckpt, args):
    symlink_force(os.path.abspath('{}.meta'.format(best_ckpt)), os.path.join(args.logdir, 'best_model.meta'))

def get_model_size(trainable_variables):
    return float(np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables])) / 1000000

