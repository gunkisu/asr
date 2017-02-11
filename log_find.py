#!/usr/bin/env python
'''Find the log file of a job produced by smart-dispatch 
based on the job id and file extension.
Can be used together with other linux commands such as tail and grep
to monitor the progress of training.'''
import os
import glob
import argparse

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
            out_file = glob.glob('{}/logs/*.{}'.format(log_dir, args.ext))[0]
            print(out_file)
            

            



