#!/usr/bin/env python
from __future__ import print_function
import os
import sys

from itertools import islice

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import namedtuple
from skiprnn2.mixer import skip_rnn_forward_parallel2, expand_output

from skiprnn2.graph_builder import match_c, match_h

from data.fuel_utils import create_ivector_test_datastream, get_uttid_stream, create_ivector_test_datastream_with_targets
from libs.utils import sync_data, skip_frames_fixed, StopWatch

import skiprnn2.utils as utils

import kaldi_io

SampleGraph = namedtuple('SampleGraph', 'step_label_probs step_action_samples step_action_probs step_last_state step_x_data init_state')

def main(_):
    print(' '.join(sys.argv), file=sys.stderr)

    args = utils.get_forward_argparser().parse_args()
    print(args, file=sys.stderr)

    sync_data(args)
    test_set = create_ivector_test_datastream(args.data_path, args.dataset, args.n_batch)

    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.n_batch) 

    with tf.Session() as sess:

        print('Loading model...', file=sys.stderr)
        model = utils.find_model(args.metafile)
        save_op = tf.train.import_meta_graph(model)
        save_op.restore(sess, model[:-5])

        writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

        _step_label_probs = sess.graph.get_tensor_by_name('step_label_probs:0')
        step_action_probs = sess.graph.get_tensor_by_name('step_action_probs:0')
        step_action_samples = sess.graph.get_tensor_by_name('step_action_samples/Multinomial:0')
        step_x_data = sess.graph.get_tensor_by_name('step_x_data:0')
        n_fast_action, = sess.graph.get_collection('n_fast_action')
        
        cstates = [op.outputs[0] for op in sess.graph.get_operations() if 'cstate' in op.name]
        hstates = [op.outputs[0] for op in sess.graph.get_operations() if 'hstate' in op.name]

        init_state = []
        for c, h in zip(cstates, hstates):
            init_state.append(tf.contrib.rnn.LSTMStateTuple(c, h))

        step_last_state = []

        last_cstates = [op.outputs[0] for op in sess.graph.get_operations() if match_c(op.name)]
        last_hstates = [op.outputs[0] for op in sess.graph.get_operations() if match_h(op.name)]
        for c, h in zip(last_cstates, last_hstates):
            step_last_state.append(tf.contrib.rnn.LSTMStateTuple(c, h))

        sample_graph = SampleGraph(_step_label_probs, step_action_samples, step_action_probs, step_last_state, step_x_data, init_state)

        writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

        print('Computing label probs...', file=sys.stderr)

        sw = StopWatch()

        for bidx, (batch, uttid_batch) in enumerate(zip(test_set.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
            orig_x, orig_x_mask, _, _ = batch
            uttid_batch, = uttid_batch

            feat_lens = orig_x_mask.sum(axis=1, dtype=np.int32)

            actions_1hot, label_probs, new_mask = skip_rnn_forward_parallel2(
                orig_x, orig_x_mask, sess, sample_graph, n_fast_action)

            seq_label_probs = expand_output(actions_1hot, orig_x_mask, new_mask, label_probs)

            for out_idx, (output, uttid) in enumerate(zip(seq_label_probs, uttid_batch)):
                valid_len = feat_lens[out_idx]
                uttid = uttid.encode('ascii')
                writer.write(uttid, np.log(output[:valid_len] + 1e-8))
            
            if args.show_progress: 
                print('.', file=sys.stderr, end='')

        print('', file=sys.stderr)
        print('Done', file=sys.stderr)
        print('Took {}'.format(sw.elapsed()), file=sys.stderr)
            
if __name__ == '__main__':
    tf.app.run()




