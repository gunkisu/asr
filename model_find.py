#!/usr/bin/env python
from __future__ import print_function

from libs.utils import log_find
import argparse

def param_extract(param, opt_line):
    items = opt_line.split(',')
    for i in items: 
        if param in i: 
            return i.strip()

    return ''

parser = argparse.ArgumentParser()
parser.add_argument('jobid')
args = parser.parse_args()

log_file = log_find(args.jobid, 'out')

save_path_line = ''

with open(log_file) as f:
    for l in f:
        if 'save_path' in l:
            save_path_line = l
            break
            
if save_path_line:
    print(param_extract('save_path', save_path_line))
