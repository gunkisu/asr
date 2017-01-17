from __future__ import print_function

import time

import os
import subprocess
from operator import attrgetter

def run_and_wait_for_output(cmd, expected_str, on='stderr'):
    proc = subprocess.Popen(cmd, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stream = attrgetter(on)(proc)

    for line in iter(stream.readline, ""):
        if expected_str in line:
            break

class StopWatch():
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time
    
    def print_elapsed(self, msg=None):
        time_diff = self.elapsed()

        if msg:
            print('{} '.format(msg), end='')
        print('Took {:.1f}s ({:.1f}m; {:.1f}h)'.format(time_diff, time_diff / 60, time_diff / 3600))
   
    def __enter__(self):
        self.reset()
        
    def __exit__(self,type,value,tb):
        self.print_elapsed()

class Rsync():
    def __init__(self, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        self.dst_dir = dst_dir

    # http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    def execute(self, cmd, shell=False):
        p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
        for stdout_line in iter(p.stdout.readline, ""):
            yield stdout_line

        p.stdout.close()
        return_code = p.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    def execute_and_print(self, cmd, shell=False):
        for line in self.execute(cmd, shell=shell):
            print(line, end="")

    def sync(self, src):
        cmd = "rsync -ahv {} {}".format(src, self.dst_dir)
        self.execute_and_print(cmd, shell=True)
