# IPython log file
from __future__ import print_function

import argparse

def parse_line(line):   
    value_part = line.split(':')[1].strip()
    value_part = value_part.split(' ')[0]
    return value_part

parser = argparse.ArgumentParser()
parser.add_argument('log_file')
args = parser.parse_args()

lines = open(args.log_file).readlines()
for l in lines: 
    if 'Model Name' in l:
        print(parse_line(l), end=' ')
    if 'Train CE' in l:
        print(parse_line(l), end=' ')
    if 'Average' in l:
        print(parse_line(l))
        
