from __future__ import print_function

import argparse
import socket
import tensorflow as tf
import subprocess
import os

from data.fuel_utils import create_ivector_datastream
from libs.utils import sync_data

def add_datapath_args(parser):
    parser.add_argument('--no-copy', action='store_true', help='Do not copy the dataset to a local disk')
    parser.add_argument('--min-after-cache', default=1024, type=int, help='Size of mini-batch')
    parser.add_argument('--no-length-sort', action='store_true', help='Do not sort the dataset by sequence lengths')
    parser.add_argument('--data-path', default='/u/songinch/song/data/speech/wsj_fbank123.h5', help='Location of the dataset')
    parser.add_argument('--tmpdir', default='/Tmp/songinch/data/speech', help='Local temporary directory to store the dataset')

def add_general_args(parser):
    parser.add_argument('--n-batch', default=16, type=int, help='Size of mini-batch')
    parser.add_argument('--device', default='gpu', help='Simply set either `cpu` or `gpu`')

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_general_args(parser)
    add_datapath_args(parser)
    
    parser.add_argument('--learning-rate', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--rl-learning-rate', default=0.01, type=float, help='Initial learning rate for RL')
    parser.add_argument('--n-epoch', default=100, type=int, help='Maximum number of epochs')
    parser.add_argument('--display-freq', default=50, type=int, help='Display frequency')
    parser.add_argument('--n-input', default=123, type=int, help='Number of RNN hidden units')
    parser.add_argument('--n-layer', default=1, type=int, help='Number of RNN hidden layers')
    parser.add_argument('--n-hidden', default=1024, type=int, help='Number of RNN hidden units')
    parser.add_argument('--n-class', default=3436, type=int, help='Number of target symbols')
    parser.add_argument('--n-embedding', default=32, type=int, help='Embedding size')
    parser.add_argument('--n-action', default=6, type=int, help='Number of actions (max skim size)')
    parser.add_argument('--n-fast-action', default=0, type=int, help='Number of steps to skip in the fast action mode')
    parser.add_argument('--base-seed', default=20170309, type=int, help='Base random seed') 
    parser.add_argument('--add-seed', default=0, type=int, help='Add this amount to the base random seed')
    parser.add_argument('--start-from-ckpt', action='store_true', help='If true, start from a ckpt')
    parser.add_argument('--logdir', default='skiprnn_test', help='Directory path to files')
    parser.add_argument('--train-dataset', default='train_si284', help='Training dataset')
    parser.add_argument('--valid-dataset', default='test_dev93', help='Validation dataset')
    parser.add_argument('--test-dataset', default='test_eval92', help='Test dataset')
    parser.add_argument('--discount-gamma', default=0.99, type=float, help='Discount factor')
    
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
    if not args.start_from_ckpt:
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

