import kaldi_io
import numpy
import h5py

f = h5py.File('/u/songinch/song/data/speech/wsj_fbank123.h5', 'a')

# Create datasets
features = f.create_dataset('features', (38230,), dtype=h5py.special_dtype(vlen=numpy.float32), maxshape=(None,))
features.dims[0].label = 'batch'
features_shapes = f.create_dataset('features_shapes', (38230,2), dtype='int32', maxshape=(None,2))
features_shapes_labels = f.create_dataset('features_shapes_labels', (2,), dtype='S7')
features.dims.create_scale(features_shapes, 'shapes')
features.dims[0].attach_scale(features_shapes)
features_shapes_labels
features_shapes_labels[...] = ['frame'.encode('utf8'), 'feature'.encode('utf8')]
features.dims.create_scale(features_shapes_labels, 'shape_labels')
features.dims[0].attach_scale(features_shapes_labels)

targets = f.create_dataset('targets', (38230,), dtype=h5py.special_dtype(vlen=numpy.int32), maxshape=(None,))
targets.dims[0].label='batch'
targets_shapes = f.create_dataset('targets_shapes', (38230,1), dtype='int32', maxshape=(None,1))
targets_shapes_labels = f.create_dataset('targets_shapes_labels', (1,), dtype='S7')
targets_shapes_labels[...] = ['frame'.encode('utf8')]
targets.dims.create_scale(targets_shapes, 'shapes')
targets.dims[0].attach_scale(targets_shapes)
targets.dims.create_scale(targets_shapes_labels, 'shape_labels')
targets.dims[0].attach_scale(targets_shapes_labels)

uttids_ds = f.create_dataset('uttids', (38230,), dtype=h5py.special_dtype(vlen=unicode), maxshape=(None,))
uttids_ds.dims[0].label = 'batch'

# Add target information
tmp = [l.strip().split(None, 1) for l in open('exp/hyperud/all_targets.txt')]
tmp_uttid = [(a, b.split()) for a, b in tmp]
    
for num_utt, (uttid, value) in enumerate(tmp_uttid):
    int_value = [int(v) for v in value]
    targets_shapes[num_utt,:] = len(int_value) 
    targets[num_utt] = numpy.asarray(int_value).ravel()

# Add uttid information
uttids = [l.split(None, 1)[0] for l open('exp/hyperud/fbank123.scp')]
for row_idx, uttid in enumerate(uttids):
    uttids_ds[row_idx] = uttid

# Add features
reader = kaldi_io.SequentialBaseFloatMatrixReader('scp:exp/hyperud/fbank123.scp')
for row_idx, (uttid, value) in enumerate(reader):
    features_shapes[row_idx,:] = value.shape
    features[row_idx] = value.ravel()

# Split information
split_dict = {
        'train_si284': {'features': (0, 37394), 'targets': (0, 37394), 'uttids': (0, 37394)},
        'train_si84': {'features': (0, 7138), 'targets': (0, 7138), 'uttids': (0, 7138)},
        'test_eval92': {'features': (37394, 37394+333), 'targets': (37394, 37394+333), 'uttids': (37394, 37394+333)},
        'test_dev93': {'features': (37394+333, 37394+333+503), 'targets': (37394+333, 37394+333+503), 'uttids': (37394+333, 37394+333+503)}
        }

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    
f.close()
