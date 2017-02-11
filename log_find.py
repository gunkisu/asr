#!/usr/bin/env python
'''Find the log file of a job launched by smart-dispatch 
based on the job id and log file extension.
It can be used together with other linux commands such as tail and grep
to monitor your jobs'''
import argparse
import os
import glob
import sys

parser = argparse.ArgumentParser()
parser.add_argument('jobid')
parser.add_argument('ext')
args = parser.parse_args()

jobid_files = glob.glob('SMART*/*/jobs_id.txt')

for jf in jobid_files:
    with open(jf) as f:
        content = f.read()
        if args.jobid in content:
            log_dir = os.path.dirname(jf)
            log_file = glob.glob('{}/logs/*.{}'.format(log_dir, args.ext))[0]
            print(log_file)

