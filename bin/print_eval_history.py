#!/usr/bin/env python
from __future__ import print_function

import numpy
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('f', help='evaluation history file')
    args = parser.parse_args()

    with numpy.load(args.f) as data:
        eval_hist = data['eval_history']
        print(eval_hist)


