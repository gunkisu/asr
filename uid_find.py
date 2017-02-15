#!/usr/bin/env python
from __future__ import print_function

from libs.utils import uid_find
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('jobid')
args = parser.parse_args()

print(uid_find(args.jobid))
