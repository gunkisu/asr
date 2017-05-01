# IPython log file
from __future__ import print_function

def parse_line(line):   
    value_part = line.split(':')[1].strip()
    value_part = value_part.split(' ')[0]
    return value_part

lines = open('train_log').readlines()
for l in lines: 
    if 'Model Name' in l:
        print(parse_line(l), end=' ')
    if 'Train CE' in l:
        print(parse_line(l), end=' ')
    if 'Average' in l:
        print(parse_line(l))
        
