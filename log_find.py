#!/usr/bin/env python
'''Find the log file of a job launched by smart-dispatch 
based on the job id and log file extension.
It can be used together with other linux commands such as tail and grep
to monitor your jobs'''

from __future__ import print_function

from libs.utils import log_find
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('jobid')
parser.add_argument('ext')
args = parser.parse_args()

print(log_find(args.jobid, args.ext))
