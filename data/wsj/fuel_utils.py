from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import Padding, FilterSources

from data.transformers import ConcatenateTransformer, TruncateTransformer, Normalize

import numpy

def uniq(seq): 
   # order preserving
   checked = []
   for e in seq:
       if e not in checked:
           checked.append(e)
   return checked

def spk_to_ids(spk_list, spks):
    return numpy.asarray([spk_list.index(s) for s in spks], dtype='int32')

def get_spk_list(speaker_ds):
    spk_list = []
    for batch in speaker_ds.get_epoch_iterator():
        assert len(batch) == 1
        for e in batch[0]:
            spk_list.append(e)

    return uniq(spk_list)

def get_feat_stream(path, which_set='test_eval92', batch_size=1, 
        norm_path=None, use_ivectors=False, truncate_ivectors=False, 
        ivector_dim=100):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, 
            batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    if norm_path: 
        data_mean_std = numpy.load(norm_path)
        base_stream = Normalize(data_stream=base_stream, 
                means=data_mean_std['mean'], stds=data_mean_std['std'])

    if use_ivectors:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'ivectors'])
        if truncate_ivectors:
            fs = TruncateTransformer(fs, 'ivectors', ivector_dim)
        # fs = ConcatenateTransformer(fs, ['features', 'ivectors'], 'features')
    else:
        fs = FilterSources(data_stream=base_stream, sources=['features'])
    return Padding(fs)


def get_uttid_stream(path, which_set='test_eval92', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, 
        batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    fs = FilterSources(data_stream=base_stream, sources=['uttids'])
    return fs

def get_spkid_stream(path, which_set='test_eval92', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    fs = FilterSources(data_stream=base_stream, sources=['spks'])
    return fs

def get_datastream(path, which_set='train_si84', batch_size=1, norm_path=None, 
        use_ivectors=False, truncate_ivectors=False, ivector_dim=100, shuffled=True):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    if shuffled:
        iterator_scheme = ShuffledScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)
    else:
        iterator_scheme = SequentialScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)

    if norm_path: 
        data_mean_std = numpy.load(norm_path)
        base_stream = Normalize(data_stream=base_stream, means=data_mean_std['mean'], stds=data_mean_std['std'])

    if use_ivectors:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'ivectors', 'targets'])
        if truncate_ivectors:
            fs = TruncateTransformer(fs, 'ivectors', ivector_dim)
        # fs = ConcatenateTransformer(fs, ['features', 'ivectors'], 'features')
    else:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'targets'])
    return Padding(fs)

def create_ivector_datastream(path, which_set='train_si84', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = ShuffledScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)
        
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)

    fs = FilterSources(data_stream=base_stream, sources=['features', 'ivectors', 'targets'])
    return Padding(fs)

def create_ivector_test_datastream(path, which_set='test_eval92', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = SequentialScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)

    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)

    fs = FilterSources(data_stream=base_stream, sources=['features', 'ivectors'])
    return Padding(fs)

