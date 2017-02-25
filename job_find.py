#!/usr/bin/env python
from __future__ import print_function

import subprocess
import glob
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slurm', action='store_true')
args = parser.parse_args()

if args.slurm:
    p = subprocess.Popen("squeue -u {}".format(os.environ['USER']), shell=True, stdout=subprocess.PIPE)
    out = p.stdout.read()
    out_lines = out.split('\n')
    for l in out_lines:
        if os.environ['USER'] not in l: continue
        items = l.split()
        if items[-4] == 'R':
            print(items[0])

else:
    p = subprocess.Popen("qstat", shell=True, stdout=subprocess.PIPE)
    out = p.stdout.read()
    out_lines = out.split('\n')

    for l in out_lines:
        if 'songinch' not in l: continue
        items = l.split()
        if items[-2] == 'R':
            print(items[0].split('.')[0])
