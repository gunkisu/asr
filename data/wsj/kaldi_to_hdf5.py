import kaldi_io
import numpy
import h5py
from fuel.datasets.hdf5 import H5PYDataset
import os.path as path

f = h5py.File('/u/songinch/song/data/speech/wsj_fbank123.h5', 'a')

exp_dir = '/u/songinch/song/kaldi/egs/wsj/s5/exp/hyperud'

# Create datasets
features = f.create_dataset('features', (38230,), dtype=h5py.special_dtype(vlen=numpy.float32), maxshape=(None,))
features.dims[0].label = 'batch'
features_shapes = f.create_dataset('features_shapes', (38230,2), dtype='int32', maxshape=(None,2))
features_shapes_labels = f.create_dataset('features_shapes_labels', (2,), dtype='S7')
features.dims.create_scale(features_shapes, 'shapes')
features.dims[0].attach_scale(features_shapes)
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

spks_ds = f.create_dataset('spks', (38230,), dtype=h5py.special_dtype(vlen=unicode), maxshape=(None,))
spks_ds.dims[0].label = 'batch'

ivectors = f.create_dataset('ivectors', (38230,), dtype=h5py.special_dtype(vlen=numpy.float32), maxshape=(None,))
ivectors.dims[0].label = 'batch'
ivectors_shapes = f.create_dataset('ivectors_shapes', (38230,2), dtype='int32', maxshape=(None,2))
ivectors_shapes_labels = f.create_dataset('ivectors_shapes_labels', (2,), dtype='S7')
ivectors.dims.create_scale(ivectors_shapes, 'shapes')
ivectors.dims[0].attach_scale(ivectors_shapes)
ivectors_shapes_labels[...] = ['frame'.encode('utf8'), 'feature'.encode('utf8')]
ivectors.dims.create_scale(ivectors_shapes_labels, 'shape_labels')
ivectors.dims[0].attach_scale(ivectors_shapes_labels)

all_targets_txt = path.join(exp_dir, 'all_targets.txt')
fbank41_scp = path.join(exp_dir, 'fbank41.scp')
utt2spk = path.join(exp_dir, 'utt2spk')

# Add target information
with open(all_targets) as f:
    tmp = [l.strip().split(None, 1) for l in f]
    tmp_uttid = [(a, b.split()) for a, b in tmp]
       
    for num_utt, (uttid, value) in enumerate(tmp_uttid):
        int_value = [int(v) for v in value]
        targets_shapes[num_utt,:] = len(int_value) 
        targets[num_utt] = numpy.asarray(int_value).ravel()

# Add uttid information
with open(fbank41_scp) as f:
    uttids = [l.strip().split(None, 1)[0] for l in f]
    for row_idx, uttid in enumerate(uttids):
        uttids_ds[row_idx] = uttid

# Add spk information
with open(utt2spk) as f:
    utt2spk = [l.strip().split(None, 1)[1] for l in f]
    for row_idx, spk in enumerate(utt2spk):
        spks_ds[row_idx] = spk

# Add features (deltas added and globally normalized on the fly)
for row_idx, (uttid, value) in enumerate(kaldi_io.SequentialBaseFloatMatrixReader('ark:add-detlas scp:exp/hyperud/fbank41.scp ark:- | apply-global-cmvn.py --global-stats=ark:exp/hyperud/cmvn-g.stats ark:- ark:-|')):
    features_shapes[row_idx,:] = value.shape
    features[row_idx] = value.ravel()

# Add ivectors
for row_idx, (uttid, value) in enumerate(kaldi_io.SequentialBaseFloatVectorReader('ark:apply-global-cmvn-vector.py ark:exp/hyperud/spk_ivectors_cmvn_g scp:exp/hyperud/spk_ivectors.scp ark:-|')):
    frame_wise_value = numpy.tile(value, (features_shapes[row_idx][0], 1))
    ivectors_shapes[row_idx,:] = frame_wise_value.shape
    ivectors[row_idx] = frame_wise_value.ravel()

f['train_si84_rand_indices'] = numpy.random.choice(37394, 7138, replace=False)
train_si84_rand_ref = f['train_si84_rand_indices'].ref


# Split information
split_dict = {
        'train_si284': {'features': (0, 37394), 'targets': (0, 37394), 'ivectors': (0, 37394), 'uttids': (0, 37394), 'spks': (0, 37394)},
        'train_si84': {'features': (0, 7138), 'targets': (0, 7138), 'ivectors': (0, 7138), 'uttids': (0, 7138), 'spks': (0, 7138)},
        'train_si84_rand': {'features': (-1, -1, train_si84_rand_ref), 'targets': (-1, -1, train_si84_rand_ref), 'ivectors': (-1, -1, train_si84_rand_ref), 'uttids': (-1, -1, train_si84_rand_ref), 'spks': (-1, -1, train_si84_rand_ref)}, 
        'test_eval92': {'features': (37394, 37394+333), 'targets': (37394, 37394+333), 'ivectors': (37394, 37394+333), 'uttids': (37394, 37394+333), 'spks': (37394, 37394+333)},
        'test_dev93': {'features': (37394+333, 37394+333+503), 'targets': (37394+333, 37394+333+503), 'ivectors': (37394+333, 37394+333+503), 'uttids': (37394+333, 37394+333+503), 'spks': (37394+333, 37394+333+503)}
        }

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    
f.close()
