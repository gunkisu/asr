#!/usr/bin/env python
from __future__ import print_function

import subprocess
import argparse
import sys
import glob

def extract_jobid(jobid_file):
    jobid = ''
    with open(jobid_file) as f:
        content = f.readlines()
        if content[-1].strip():
            jobid = content[-1].split('.')[0]

    return jobid

def extract_all_jobids(log_dir='SMART_DISPATCH_LOGS'):
    jobid_files = glob.glob('{}/*/jobs_id.txt'.format(log_dir))
    return [ extract_jobid(jf) for jf in jobid_files ]

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true')
args = parser.parse_args()


if args.all:
    jobids = extract_all_jobids()
    for ji in sorted(jobids): 
        if ji: print(ji)
    sys.exit(0) 

p = subprocess.Popen("qstat", shell=True, stdout=subprocess.PIPE)
out = p.stdout.read()
out_lines = out.split('\n')

for l in out_lines:
    if 'songinch' not in l: continue
    items = l.split()
    if items[-2] == 'R':
        print(items[0].split('.')[0])
