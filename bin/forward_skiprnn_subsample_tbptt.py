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
from libs.utils import sync_data, skip_frames_fixed2, StopWatch
from skiprnn2.mixer import gen_zero_state, feed_init_state, fixed_skip_forward

import skiprnn2.mixer as mixer

from skiprnn2.graph_builder import match_c_fw, match_h_fw

import skiprnn2.utils as utils
import kaldi_io

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

        g_seq_label_probs = sess.graph.get_tensor_by_name('seq_label_probs:0')
        g_seq_x_data = sess.graph.get_tensor_by_name('seq_x_data:0')
        g_seq_x_mask = sess.graph.get_tensor_by_name('seq_x_mask:0')

        cstates_fw = [op.outputs[0] for op in sess.graph.get_operations() if 'cstate' in op.name and 'bw' not in op.name]
        hstates_fw = [op.outputs[0] for op in sess.graph.get_operations() if 'hstate' in op.name and 'bw' not in op.name]

        init_state_fw = []
        for c, h in zip(cstates_fw, hstates_fw):
            init_state_fw.append(tf.contrib.rnn.LSTMStateTuple(c, h))

        cstates_bw = [op.outputs[0] for op in sess.graph.get_operations() if 'cstate' in op.name and 'bw' in op.name]
        hstates_bw = [op.outputs[0] for op in sess.graph.get_operations() if 'hstate' in op.name and 'bw' in op.name]

        init_state_bw = []
        for c, h in zip(cstates_bw, hstates_bw):
            init_state_bw.append(tf.contrib.rnn.LSTMStateTuple(c, h))

        g_output_state_fw = []
        last_cstates = [op.outputs[0] for op in sess.graph.get_operations() if match_c_fw(op.name)]
        last_hstates = [op.outputs[0] for op in sess.graph.get_operations() if match_h_fw(op.name)]
        for c, h in zip(last_cstates, last_hstates):
            g_output_state_fw.append(tf.contrib.rnn.LSTMStateTuple(c, h))
        
        n_skip, = sess.graph.get_collection('n_skip')
        n_step, = sess.graph.get_collection('n_step')
        n_hidden, = sess.graph.get_collection('n_hidden')
        n_layer, = sess.graph.get_collection('n_layer')
        n_class, = sess.graph.get_collection('n_class')
#        n_step = 22; n_hidden = 256; n_layer = 2; n_class = 3436

        writer = kaldi_io.BaseFloatMatrixWriter(args.wxfilename)

        print('Computing label probs...', file=sys.stderr)

        sw = StopWatch()
        
        for bidx, (batch, uttid_batch) in enumerate(zip(test_set.get_epoch_iterator(), uttid_stream.get_epoch_iterator())):
            orig_x, orig_x_mask, _, _ = batch
            uttid_batch, = uttid_batch

            feat_lens = orig_x_mask.sum(axis=1)

            sub_batch = skip_frames_fixed2([orig_x, orig_x_mask], n_skip+1, return_first=True)
            skip_x, skip_x_mask = sub_batch
            n_batch = skip_x.shape[0]

            prev_state_fw = np.zeros([n_batch, n_layer, 2, n_hidden])
            prev_state_bw = np.zeros([n_batch, n_layer, 2, n_hidden])

            seq_label_probs_list = []

            for win_idx, win in enumerate(utils.win_iter(sub_batch, n_step), start=1):
                x, x_mask = win

                feed_dict={g_seq_x_data: x, g_seq_x_mask: x_mask}

                mixer.feed_prev_state(feed_dict, init_state_fw, prev_state_fw)
                mixer.feed_prev_state(feed_dict, init_state_bw, prev_state_bw)

                _seq_label_probs, output_state_fw = \
                    sess.run([g_seq_label_probs, g_output_state_fw], feed_dict=feed_dict)

                output_state_fw = np.transpose(np.asarray(output_state_fw), [2,0,1,3])
                mixer.update_prev_state(prev_state_fw, output_state_fw)
                
                # shape of seq_label_probs: [n_batch * n_seq, n_class]
                seq_label_probs_list.append(_seq_label_probs.reshape([n_batch, -1, n_class]))
            
            seq_label_probs = np.concatenate(seq_label_probs_list, axis=1)
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

