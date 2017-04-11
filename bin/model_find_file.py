#!/usr/bin/env python
from __future__ import print_function

import argparse

def param_extract(param, opt_line):
    items = opt_line.split(',')
    for i in items: 
        if param in i: 
            return i.strip()

    return ''

def extract_num_params(num_params_line):
    items = num_params_line.split(':')

    return items[-1].strip()

parser = argparse.ArgumentParser()
parser.add_argument('logfile')
args = parser.parse_args()

log_file = args.logfile

save_path_line = ''
num_params_line = ''

with open(log_file) as f:
    for l in f:
        if 'save_path' in l:
            save_path_line = l
        if 'Number of parameters' in l:
            num_params_line = l

if save_path_line:
    print(param_extract('save_path', save_path_line), end=' ')

if num_params_line:
    print(extract_num_params(num_params_line))
