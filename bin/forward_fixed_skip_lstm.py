#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf

import glob

from collections import namedtuple, OrderedDict

from data.fuel_utils import create_ivector_test_datastream, get_uttid_stream
from libs.utils import sync_data, skip_frames_fixed, StopWatch

from skiprnn.mixer import gen_zero_state, feed_init_state


from itertools import islice

import kaldi_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n-batch', 64, 'Size of mini-batch')
flags.DEFINE_string('device', 'gpu', 'Simply set either `cpu` or `gpu`')
flags.DEFINE_boolean('no-copy', True, '') # test set is typically small
flags.DEFINE_string('tmpdir', '/Tmp/songinch/data/speech', '')
flags.DEFINE_string('data-path', '/u/songinch/song/data/speech/wsj_fbank123.h5', '')
flags.DEFINE_string('dataset', 'test_dev93', '')
flags.DEFINE_string('wxfilename', 'ark:-', '')
flags.DEFINE_string('metafile', 'best_model.ckpt-1000.meta', '')

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

        _seq_label_probs = sess.graph.get_tensor_by_name('seq_label_probs:0')
        seq_x_data = sess.graph.get_tensor_by_name('seq_x_data:0')
        seq_x_mask = sess.graph.get_tensor_by_name('seq_x_mask:0')

        cstates = [op.outputs[0] for op in sess.graph.get_operations() if 'cstate' in op.name]
        hstates = [op.outputs[0] for op in sess.graph.get_operations() if 'hstate' in op.name]

        init_state = []
        for c, h in zip(cstates, hstates):
            init_state.append(tf.contrib.rnn.LSTMStateTuple(c, h))
        n_hidden, = sess.graph.get_collection('n_hidden')

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

            zero_state = gen_zero_state(n_batch, n_hidden)

            feed_dict={seq_x_data: x, seq_x_mask: x_mask}
            feed_init_state(feed_dict, init_state, zero_state)

            seq_label_probs, = sess.run([_seq_label_probs], feed_dict=feed_dict)

            seq_label_probs = seq_label_probs.reshape([n_batch, n_seq, -1])
            seq_label_probs = seq_label_probs.repeat(n_skip+1, axis=1)

            for out_idx, (output, uttid) in enumerate(zip(seq_label_probs, uttid_batch)):
                valid_len = int(feat_lens[out_idx])
                uttid = uttid.encode('ascii')
                writer.write(uttid, np.log(output[:valid_len] + 1e-8))

            print('.', file=sys.stderr, end='')

        print('', file=sys.stderr)
        print('Done', file=sys.stderr)
        print('Took {}'.format(sw.elapsed()), file=sys.stderr)

if __name__ == '__main__':
    tf.app.run()

