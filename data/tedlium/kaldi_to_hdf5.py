import kaldi_io
import numpy
import h5py
from fuel.datasets.hdf5 import H5PYDataset
import os.path as path

f = h5py.File('/u/songinch/song/data/speech/tedlium_fbank123_out4174.h5', 'a')
exp_dir = '/u/songinch/song/kaldi/egs/tedlium/s5_r2/exp/model_adapt'

# Create datasets
features = f.create_dataset('features', (94145,), dtype=h5py.special_dtype(vlen=numpy.float32), maxshape=(None,))
features.dims[0].label = 'batch'
features_shapes = f.create_dataset('features_shapes', (94145,2), dtype='int32', maxshape=(None,2))
features_shapes_labels = f.create_dataset('features_shapes_labels', (2,), dtype='S7')
features.dims.create_scale(features_shapes, 'shapes')
features.dims[0].attach_scale(features_shapes)
features_shapes_labels[...] = ['frame'.encode('utf8'), 'feature'.encode('utf8')]
features.dims.create_scale(features_shapes_labels, 'shape_labels')
features.dims[0].attach_scale(features_shapes_labels)

targets = f.create_dataset('targets', (94145,), dtype=h5py.special_dtype(vlen=numpy.int32), maxshape=(None,))
targets.dims[0].label='batch'
targets_shapes = f.create_dataset('targets_shapes', (94145,1), dtype='int32', maxshape=(None,1))
targets_shapes_labels = f.create_dataset('targets_shapes_labels', (1,), dtype='S7')
targets_shapes_labels[...] = ['frame'.encode('utf8')]
targets.dims.create_scale(targets_shapes, 'shapes')
targets.dims[0].attach_scale(targets_shapes)
targets.dims.create_scale(targets_shapes_labels, 'shape_labels')
targets.dims[0].attach_scale(targets_shapes_labels)

uttids_ds = f.create_dataset('uttids', (94145,), dtype=h5py.special_dtype(vlen=unicode), maxshape=(None,))
uttids_ds.dims[0].label = 'batch'

spks_ds = f.create_dataset('spks', (94145,), dtype=h5py.special_dtype(vlen=unicode), maxshape=(None,))
spks_ds.dims[0].label = 'batch'

ivectors = f.create_dataset('ivectors', (94145,), dtype=h5py.special_dtype(vlen=numpy.float32), maxshape=(None,))
ivectors.dims[0].label = 'batch'
ivectors_shapes = f.create_dataset('ivectors_shapes', (94145,2), dtype='int32', maxshape=(None,2))
ivectors_shapes_labels = f.create_dataset('ivectors_shapes_labels', (2,), dtype='S7')
ivectors.dims.create_scale(ivectors_shapes, 'shapes')
ivectors.dims[0].attach_scale(ivectors_shapes)
ivectors_shapes_labels[...] = ['frame'.encode('utf8'), 'feature'.encode('utf8')]
ivectors.dims.create_scale(ivectors_shapes_labels, 'shape_labels')
ivectors.dims[0].attach_scale(ivectors_shapes_labels)

all_targets_txt = path.join(exp_dir, 'all_targets.txt')
all_fbank41_scp = path.join(exp_dir, 'all_fbank41.scp')
all_utt2spk = path.join(exp_dir, 'all_utt2spk')
cmvn_stats = path.join(exp_dir, 'cmvn.stats')
all_spk_ivectors_scp = path.join(exp_dir, 'all_spk_ivectors.scp')
spk_ivector_cmvn_stats = path.join(exp_dir, 'spk_ivector_cmvn.stats')

# Add target information
with open(all_targets_txt) as f:
    tmp = [l.strip().split(None, 1) for l in f]
    tmp_uttid = [(a, b.split()) for a, b in tmp]
       
    for num_utt, (uttid, value) in enumerate(tmp_uttid):
        int_value = [int(v) for v in value]
        targets_shapes[num_utt,:] = len(int_value) 
        targets[num_utt] = numpy.asarray(int_value).ravel()

# Add uttid information
with open(all_fbank41_scp) as f:
    uttids = [l.strip().split(None, 1)[0] for l in f]
    for row_idx, uttid in enumerate(uttids):
        uttids_ds[row_idx] = uttid

# Add spk information
with open(all_utt2spk) as f:
    utt2spk = [l.strip().split(None, 1)[1] for l in f]
    for row_idx, spk in enumerate(utt2spk):
        spks_ds[row_idx] = spk

feat = 'ark:add-deltas scp:{} ark:- | apply-global-cmvn.py --global-stats=ark:{} ark:- ark:-|'.format(all_fbank41_scp, cmvn_stats)

# Add features (deltas added and globally normalized on the fly)
for row_idx, (uttid, value) in enumerate(kaldi_io.SequentialBaseFloatMatrixReader(feat)):
    features_shapes[row_idx,:] = value.shape
    features[row_idx] = value.ravel()

ivector= 'ark:apply-global-cmvn-vector.py ark:{} scp:{} ark:-|'.format(spk_ivector_cmvn_stats, all_spk_ivectors_scp)

# Add ivectors
for row_idx, (uttid, value) in enumerate(kaldi_io.SequentialBaseFloatVectorReader(ivector)):
    frame_wise_value = numpy.tile(value, (features_shapes[row_idx][0], 1))
    ivectors_shapes[row_idx,:] = frame_wise_value.shape
    ivectors[row_idx] = frame_wise_value.ravel()

#f['train_si84_rand_indices'] = numpy.random.choice(37394, 7138, replace=False)
#train_si84_rand_ref = f['train_si84_rand_indices'].ref

# Split information

#     498 dev_fbank41_with_targets.scp
#    1147 test_fbank41_with_targets.scp
#   92500 train_fbank41_with_targets.scp
#   94145 total


split_dict = {
        'train': {'features': (0, 92500), 'targets': (0, 92500), 'ivectors': (0, 92500), 'uttids': (0, 92500), 'spks': (0, 92500)},
        'dev': {'features': (92500, 92500+498), 'targets': (92500, 92500+498), 'ivectors': (92500, 92500+498), 'uttids': (92500, 92500+498), 'spks': (92500, 92500+498)},
        'test': {'features': (92500+498, 92500+498+1147), 'targets': (92500+498, 92500+498+1147), 'ivectors': (92500+498, 92500+498+1147), 'uttids': (92500+498, 92500+498+1147), 'spks': (92500+498, 92500+498+1147)}
        }

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    
f.close()
