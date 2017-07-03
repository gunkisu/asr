#!/usr/bin/env python
from __future__ import print_function
import os
import sys
sys.path.insert(0, '..')

from itertools import islice

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple
from skiprnn.mixer import skip_rnn_forward_parallel2, expand_output

from data.fuel_utils import create_ivector_test_datastream, get_uttid_stream, create_ivector_test_datastream_with_targets
from libs.utils import sync_data, skip_frames_fixed, StopWatch

import kaldi_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n-batch', 64, 'Size of mini-batch')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_boolean('no-copy', True, '')
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data-path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('dataset', 'test_dev93', '')
flags.DEFINE_string('wxfilename', 'ark:-', '')
flags.DEFINE_string('metafile', 'best_model.ckpt-1000.meta', '')

SampleGraph = namedtuple('SampleGraph', 'step_label_probs step_action_samples step_action_probs step_last_state step_x_data prev_states')

def initial_states(batch_size, n_hidden):
    init_state = np.zeros([2, batch_size, n_hidden], dtype=np.float32)
    return init_state

def main(_):
    print(' '.join(sys.argv), file=sys.stderr)
    args = FLAGS
    print(args.__flags, file=sys.stderr)

    sync_data(args)
    test_set = create_ivector_test_datastream(args.data_path, args.dataset, args.n_batch)

    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.n_batch) 

    with tf.Session() as sess:

        print('Loading model...', file=sys.stderr)
        save_op = tf.train.import_meta_graph(args.metafile)
        save_op.restore(sess, args.metafile[:-5])

        writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

        _step_label_probs = sess.graph.get_tensor_by_name('step_label_probs:0')
        step_action_probs = sess.graph.get_tensor_by_name('step_action_probs:0')
        step_action_samples = sess.graph.get_tensor_by_name('step_action_samples/Multinomial:0')
        step_x_data = sess.graph.get_tensor_by_name('step_x_data:0')
        prev_states = sess.graph.get_tensor_by_name('prev_states:0')
        step_last_state = sess.graph.get_tensor_by_name('one_step_stack:0')
        fast_action, n_fast_action = sess.graph.get_collection('fast_action')

        sample_graph = SampleGraph(_step_label_probs, step_action_samples, step_action_probs, step_last_state, step_x_data, prev_states)

        writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

        print('Computing label probs...', file=sys.stderr)

        sw = StopWatch()
        
        # debugging
        fixed_skip = True

        for bidx, (batch, uttid_batch) in enumerate(zip(test_set.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
            orig_x, orig_x_mask, _, _, y, _ = batch
            uttid_batch, = uttid_batch

            feat_lens = orig_x_mask.sum(axis=1, dtype=np.int32)

            actions_1hot, label_probs, new_mask = skip_rnn_forward_parallel2(
                orig_x, orig_x_mask, sess, sample_graph, fast_action, n_fast_action, fixed_skip)

            seq_label_probs = expand_output(actions_1hot, orig_x_mask, new_mask, label_probs)

            for out_idx, (output, uttid) in enumerate(zip(seq_label_probs, uttid_batch)):
                valid_len = feat_lens[out_idx]
                uttid = uttid.encode('ascii')
                writer.write(uttid, np.log(output[:valid_len] + 1e-8))

            print('.', file=sys.stderr, end='')

        print('', file=sys.stderr)
        print('Done', file=sys.stderr)
        print('Took {}'.format(sw.elapsed()), file=sys.stderr)
            
if __name__ == '__main__':
    tf.app.run()




