import argparse
import os

def add_deep_lstm_params(parser):
    parser.add_argument('--batch-size', default=2, help='batch size', type=int)
    parser.add_argument('--n-hidden', default=512, help='number of hidden units', type=int)
    parser.add_argument('--learn-rate', default=0.002, help='learning rate', type=float)
    parser.add_argument('--use-impl-type', type=str, default='base')
    parser.add_argument('--data-path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--n-input', help='input dimension', default=123, type=int)
    parser.add_argument('--n-input-embed', help=' ', default=256, type=int)
    parser.add_argument('--n-class', help='output dimension', default=3436, type=int)
    parser.add_argument('--n-output-embed', help=' ', default=256, type=int)
    parser.add_argument('--num-epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--start-from-ckpt', action="store_true")
    parser.add_argument('--use-slope-anneal', action='store_true')
    parser.add_argument('--log-freq', help='how ferquently to display progress', default=50, type=int)
    parser.add_argument('--train-dataset', help='dataset for training', default='train_si284')
    parser.add_argument('--valid-dataset', help='dataset for validation', default='test_dev93')
    parser.add_argument('--test-dataset', help='dataset for test', default='test_eval92')
    parser.add_argument('--no-copy', help='do not copy data from NFS to local machine', action='store_true')
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally', default='/Tmp/songinch/data/speech')
    parser.add_argument('--log-dir', help=' ', default='hmrnn_train_log')



def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_deep_lstm_params(parser)
    return parser

def get_save_path(args):
    return 'hmrnn_test'

