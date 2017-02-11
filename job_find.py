#!/usr/bin/env python
from __future__ import print_function

import subprocess

p = subprocess.Popen("qstat", shell=True, stdout=subprocess.PIPE)
out = p.stdout.read()

out_lines = out.split('\n')

for l in out_lines:
    if 'songinch' not in l: continue
    items = l.split()
    if items[-2] == 'R':
        print(items[0].split('.')[0])
