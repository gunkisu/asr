# IPython log file

import h5py
f =h5py.File('/u/songinch/song/data/speech/wsj_fbank123.h5', 'a')
import numpy
features = f.create_dataset('features', (38230,), dtype=h5py.special_dtype(vlen=numpy.float32), maxshape=(None,))
features.dims[0].label = 'batch'
targets = f.create_dataset('targets', (37394,), dtype=h5py.special_dtype(vlen=numpy.int32), maxshape=(None,))
targets.dims[0].label='batch'
features_shapes = f.create_dataset('features_shapes', (38230,2), dtype='int32', maxshape=(None,2))
features_shapes_labels = f.create_dataset('features_shapes_labels', (2,), dtype='S7')
features.dims.create_scale(features_shapes, 'shapes')
features.dims[0].attach_scale(features_shapes)
features_shapes_labels
features_shapes_labels[...] = ['frame'.encode('utf8'), 'feature'.encode('utf8')]
features.dims.create_scale(features_shapes_labels, 'shape_labels')
features.dims[0].attach_scale(features_shapes_labels)
targets_shapes_labels = f.create_dataset('targets_shapes_labels', (1,), dtype='S7')
targets_shapes_labels[...] = ['frame'.encode('utf8')]
targets.dims.create_scale(targets_shapes, 'shapes')
targets.dims[0].attach_scale(targets_shapes)
targets.dims.create_scale(targets_shapes_labels, 'shape_labels')
targets.dims[0].attach_scale(targets_shapes_labels)

alignments = list(open('exp/hyperud/all_targets.txt'))
tmp = [l.strip().split(None, 1) for l in open('exp/hyperud/all_targets.txt')]
tmp_uttid = [(a, b.split()) for a, b in tmp]
    
for num_utt, (uttid, value) in enumerate(tmp_uttid):
    targets_shapes[num_utt,:] = len(value) 
    targets[num_utt] = numpy.asarray(value).ravel()
    
f.close()
