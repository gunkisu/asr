from __future__ import print_function

import time
import random
import numpy as np
import os
import subprocess
from operator import attrgetter

import argparse
import glob
import sys

import pickle

from libs.param_utils import get_model_param_values, get_update_params_values, set_model_param_value

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

def show_status2(save_path, ce_frame, batch_idx, batch_size, epoch_idx, avg_z_1_3d):
    model = save_path.split('/')[-1]
    print('--')
    print('Model Name: {} (Epoch {})'.format(model, epoch_idx))
    print('Train CE: {} (batch {}, {} examples so far): '.format(ce_frame, batch_idx, batch_idx*batch_size))
    print('Average z_1_3d: {}'.format(avg_z_1_3d))


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

def sync_data(args):
    sw = StopWatch()
    if not args.no_copy:
        print('Copying data to local machine...')
        sw.reset()
        rsync = Rsync(args.tmpdir)
        rsync.sync(args.data_path)
        args.data_path = os.path.join(args.tmpdir, os.path.basename(args.data_path))
        sw.print_elapsed()

def load_or_init_model(network_params, args):
    pretrain_update_params_val = None
    pretrain_total_epoch_cnt = 0
 
    if args.reload_model:
        print('Loading the model: {}'.format(args.reload_model))
        with open(args.reload_model, 'rb') as f:
            pretrain_network_params_val,  pretrain_update_params_val, \
                    pretrain_total_epoch_cnt = pickle.load(f)

        set_model_param_value(network_params, pretrain_network_params_val)
 
    return pretrain_update_params_val, pretrain_total_epoch_cnt

def find_reload_model(args):
    if not args.reload_model and not args.no_reload:
        reload_path = args.save_path + '_last_model.pkl'

        if os.path.exists(reload_path):
            print('Previously trained model detected: {}'.format(reload_path))
            print('Training continues')
            args.reload_model = reload_path


def gen_win(batch, win_size, right_context=0):
    input_data, input_mask, ivector_data, ivector_mask, target_data, target_mask = batch
    
    n_batch, n_seq, n_feat = input_data.shape

    for i in range(0, n_seq, win_size):
        from_idx = i
        to_idx = i+win_size+right_context
        
        yield (input_data[:,from_idx:to_idx,:], input_mask[:,from_idx:to_idx], \
            ivector_data[:,from_idx:to_idx,:], ivector_mask[:,from_idx:to_idx],  \
            target_data[:,from_idx:to_idx], target_mask[:,from_idx:to_idx])
   
    return

def gen_win_test(batch, win_size, right_context=0):
    input_data, input_mask, ivector_data, ivector_mask = batch
    
    n_batch, n_seq, n_feat = input_data.shape

    for i in range(0, n_seq, win_size):
        from_idx = i
        to_idx = i+win_size+right_context
        
        yield (input_data[:,from_idx:to_idx,:], input_mask[:,from_idx:to_idx], \
            ivector_data[:,from_idx:to_idx,:], ivector_mask[:,from_idx:to_idx])
   
    return


def skip_frames(batch, every_n, random_choice=False):
    new_batch = []

    for src_data in batch:
        new_src_data = []
        for ex in src_data:
            n_seq = ex.shape[0]
            if random_choice:
                idx_list = [min(i+random.randint(0, every_n-1), n_seq-1) for i in range(0, n_seq, every_n)]
                if len(ex.shape) == 1:

                    new_src_data.append(ex[idx_list])
                else:
                    new_src_data.append(ex[idx_list,:])
            else:
                if len(ex.shape) == 1: 
                    new_src_data.append(ex[::every_n])
                else:
                    new_src_data.append(ex[::every_n,:])    

        new_batch.append(new_src_data)
    return new_batch

def skip_frames_fixed(batch, every_n, return_first=False):
    assert every_n > 0

    if every_n == 1:
        yield batch

    sub_seq_iter = None
    if return_first:
        sub_seq_iter = range(1)
    else:
        sub_seq_iter = range(every_n)

    for start_idx in sub_seq_iter:
        new_batch = []

        for src_data in batch:
            new_src_data = []
            for ex in src_data:
                assert not ex.shape[0] < every_n
                
                if len(ex.shape) == 1: 
                    new_src_data.append(ex[start_idx::every_n])

                else:
                    new_src_data.append(ex[start_idx::every_n,:])    

            new_batch.append(new_src_data)
        yield new_batch

def skip_frames_random(batch, every_n):
    new_batch = []

    for src_data in batch:
        new_src_data = []
        for ex in src_data:
            n_seq = ex.shape[0]

            idx_list = [min(i+random.randint(0, every_n-1), n_seq-1) for i in range(0, n_seq, every_n)]
            if len(ex.shape) == 1:

                new_src_data.append(ex[idx_list])
            else:
                new_src_data.append(ex[idx_list,:])

        new_batch.append(new_src_data)
    return new_batch

def split(row):
    segs = []
    for i, item in enumerate(row):
        if i == 0:
            seg = [item]
            continue

        if item > 0:
            segs.append(seg)
            seg = [item]
        else:
            seg.append(item)

    segs.append(seg)
    
    return segs

def seg_len_info(z_1_3d):
    a_list = []
    for row in z_1_3d:
        a_list.append(split(row))
    
    lens = []
    for row in a_list:
        lens.append([len(seg) for seg in row])
           
    return lens

def compress_batch(batch, z_1_3d):
    z_1_3d[:,0] = 1.0 # Always keep the first frame
    z_1_3d_bmask = z_1_3d > 0
    filtered = batch[z_1_3d_bmask]
    seq_lens = z_1_3d.sum(axis=1)

    compressed = []
    sidx = 0
    for sl in seq_lens:
        compressed.append(filtered[sidx:sidx+sl])
        sidx += sl
    
    return pad_batch(compressed)

def uncompress_batch(compressed, seg_len_info):
    n_batch, n_seq = compressed.shape[0], compressed.shape[1]
    rest_shape = compressed.shape[2:]

    max_seq_len = sum(seg_len_info[0])
    uncompressed = np.zeros( (n_batch,max_seq_len) + rest_shape)
    
    for i, len_info in enumerate(seg_len_info):
        row = compressed[i]
        
        sidx = 0
        for j, l in enumerate(len_info):
            uncompressed[i,sidx:sidx+l] = row[j]
            sidx += l
    return uncompressed

def pad_batch(batch):
    shapes = [b.shape for b in batch]

    seq_lens = [shape[0] for shape in shapes]
    max_seq_len = max(seq_lens)
    rest_shape = shapes[0][1:]
    if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")

    dtype = batch[0].dtype

    padded_batch = np.zeros( (len(batch), max_seq_len) + rest_shape, dtype=dtype)

    for i, b in enumerate(batch):
        padded_batch[i,:len(b)] = b

    return padded_batch