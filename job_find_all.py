#!/usr/bin/env python
from __future__ import print_function

import glob

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

jobids = extract_all_jobids()
for ji in sorted(jobids): 
    if ji: print(ji)
