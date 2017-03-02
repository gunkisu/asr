#!/usr/bin/env python
from __future__ import print_function

import subprocess
import glob
import argparse
import os

def extract_jobid(jobid_file):
    jobid = ''
    with open(jobid_file) as f:
        for l in f:
            if 'helios' in l:
                jobid = l.split('.')[0]

    # There can be multiple job ids because of resumption.
    # The last job id will be returned.

    return jobid

def extract_all_jobids(log_dir='SMART_DISPATCH_LOGS'):
    jobid_files = glob.glob('{}/*/jobs_id.txt'.format(log_dir))
    return [extract_jobid(jf) for jf in jobid_files]

def extract_all_slurm_jobids(log_dir='.'):
    jobid_files = glob.glob('{}/slurm*.out'.format(log_dir))
    return [os.path.basename(jf).split('.')[0][6:] for jf in jobid_files]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--all', action='store_true')
args = parser.parse_args()

if args.slurm:
    if args.all:
        jobids = extract_all_slurm_jobids()
        for ji in sorted(jobids): 
            if ji: print(ji)

    else:
        p = subprocess.Popen("squeue -u {}".format(os.environ['USER']), shell=True, stdout=subprocess.PIPE)
        out = p.stdout.read()
        out_lines = out.split('\n')
        for l in out_lines:
            if os.environ['USER'] not in l: continue
            if 'bash' in l: continue # sinter
            items = l.split()
            if items[-4] == 'R':
                print(items[0])

else:
    if args.all:
        jobids = extract_all_jobids()
        for ji in sorted(jobids): 
            if ji: print(ji)
    else:
        p = subprocess.Popen("qstat", shell=True, stdout=subprocess.PIPE)
        out = p.stdout.read()
        out_lines = out.split('\n')

        for l in out_lines:
            if 'songinch' not in l: continue
            items = l.split()
            if items[-2] == 'R':
                print(items[0].split('.')[0])
