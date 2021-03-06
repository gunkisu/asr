#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf
import glob
from collections import namedtuple, OrderedDict
from itertools import islice

from data.fuel_utils import create_ivector_test_datastream, get_uttid_stream
from libs.utils import sync_data, skip_frames_fixed, StopWatch
from skiprnn2.mixer import gen_zero_state, feed_init_state, fixed_skip_forward

from skiprnn2.graph_builder import match_c, match_h

import skiprnn2.utils as utils
import kaldi_io

TestGraph = namedtuple('TestGraph', 'step_x_data init_state step_last_state step_label_probs')

if __name__ == '__main__':
    print(' '.join(sys.argv), file=sys.stderr)

    args = utils.get_forward_argparser().parse_args()
    print(args, file=sys.stderr)

    sync_data(args)
    test_set = create_ivector_test_datastream(args.data_path, args.dataset, args.n_batch)

    uttid_stream = get_uttid_stream(args.data_path, args.dataset, args.n_batch) 

    with tf.Session() as sess:

        model = utils.find_model(args.metafile)
        print('Loading model: {}'.format(model), file=sys.stderr)

        save_op = tf.train.import_meta_graph(model)
        save_op.restore(sess, model[:-5])

        step_label_probs = sess.graph.get_tensor_by_name('step_label_probs:0')
        step_x_data = sess.graph.get_tensor_by_name('step_x_data:0')

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

        tg = TestGraph(step_x_data, init_state, step_last_state, step_label_probs)        
        
        n_skip, = sess.graph.get_collection('n_skip')

        writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

        print('Computing label probs...', file=sys.stderr)

        sw = StopWatch()
        
        for bidx, (batch, uttid_batch) in enumerate(zip(test_set.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
            orig_x, orig_x_mask, _, _ = batch
            uttid_batch, = uttid_batch

            feat_lens = orig_x_mask.sum(axis=1)

            sub_batch, = list(skip_frames_fixed([orig_x, orig_x_mask], n_skip+1, return_first=True))
            x, x_mask = sub_batch
            n_batch, n_seq, _ = x.shape

            seq_label_probs, = fixed_skip_forward(x, x_mask, sess, tg)

            seq_label_probs = seq_label_probs.reshape([n_batch, n_seq, -1])
            seq_label_probs = seq_label_probs.repeat(n_skip+1, axis=1)

            for out_idx, (output, uttid) in enumerate(zip(seq_label_probs, uttid_batch)):
                valid_len = int(feat_lens[out_idx])
                uttid = uttid.encode('ascii')
                writer.write(uttid, np.log(output[:valid_len] + 1e-8))
            
            if args.show_progress: 
                print('.', file=sys.stderr, end='')

        print('', file=sys.stderr)
        print('Done', file=sys.stderr)
        print('Took {}'.format(sw.elapsed()), file=sys.stderr)
