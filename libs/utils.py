from __future__ import print_function

import time

import os
import subprocess
from operator import attrgetter

import argparse
import glob
import sys

import pickle

from libs.lasagne_libs.utils import get_model_param_values, get_update_params_values

def run_and_wait_for_output_on_stderr(cmd, expected_str):
    proc = subprocess.Popen(cmd, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    for line in iter(proc.stderr.readline, ""):
        if expected_str in line:
            break
    
    return proc 

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

def save_network(network_params, trainer_params, epoch_cnt, save_path):
    with open(save_path, 'wb') as f:
        cur_network_params_val = get_model_param_values(network_params)
        cur_trainer_params_val = get_update_params_values(trainer_params)
        pickle.dump([cur_network_params_val, cur_trainer_params_val, epoch_cnt],
                    f)

def save_eval_history(eval_history, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(eval_history, f)

def symlink_force(src, link_name):
    if os.path.exists(link_name):
        os.remove(link_name)
    os.symlink(src, link_name)
    
def best_fer(eval_history):
    acopy = list(eval_history)
    acopy.sort(key=attrgetter('valid_fer'))
    return acopy[0].valid_fer


def show_status(save_path, ce_frame, network_grads_norm, batch_idx, batch_size, epoch_idx):
    model = save_path.split('/')[-1]
    print('--')
    print('Model Name: {} (Epoch {})'.format(model, epoch_idx))
    print('Train CE {} (batch {}, {} examples so far): '.format(ce_frame, batch_idx, batch_idx*batch_size))
    print('Gradient Norm: {}'.format(network_grads_norm))


def log_find(jobid, ext, log_dir='SMART_DISPATCH_LOGS'):
    '''Find the log file of a job launched by smart-dispatch 
    based on the job id and log file extension.
    It can be used together with other linux commands such as tail and grep
    to monitor your jobs'''

    jobid_files = glob.glob('{}/*/jobs_id.txt'.format(log_dir))

    for jf in jobid_files:
        with open(jf) as f:
            content = f.read()
            if jobid in content:
                log_file = glob.glob('{}/logs/*.{}'.format(os.path.dirname(jf), ext))
                if log_file: return log_file[0]
                else: return ''
    return ''

def uid_find(jobid, log_dir='SMART_DISPATCH_LOGS'):
    '''Find the directory of a job given its id. The directory name can be used
    to resume the job by smart-dispatch'''

    jobid_files = glob.glob('{}/*/jobs_id.txt'.format(log_dir))

    for jf in jobid_files:
        with open(jf) as f:
            content = f.read()
            if jobid in content:
                return os.path.dirname(jf)

    return ''
