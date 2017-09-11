#!/usr/bin/env python
from __future__ import print_function
import argparse

from skiprnn2.utils import find_model_iter_nums

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('exp_dir')
parser.add_argument('--best', action='store_true')
args = parser.parse_args()

iters = find_model_iter_nums(args.exp_dir, args.best)
iters = [str(i) for i in iters]
print(' '.join(iters))
