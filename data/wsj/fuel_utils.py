from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import Padding, FilterSources

from data.transformers import ConcatenateTransformer, TruncateTransformer


def get_feat_stream(path, which_set='test_eval92', batch_size=1, use_ivectors=False):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    if use_ivectors:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'ivectors'])
        fs = ConcatenateTransformer(fs, ['features', 'ivectors'], 'features')
    else:
        fs = FilterSources(data_stream=base_stream, sources=['features'])
    return Padding(fs)

def get_uttid_stream(path, which_set='test_eval92', batch_size=1):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = SequentialScheme(examples=wsj_dataset.num_examples, batch_size=batch_size)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)
    fs = FilterSources(data_stream=base_stream, sources=['uttids'])
    return fs

def get_datastream(path, which_set='train_si84', batch_size=1, use_ivectors=False, truncate_ivectors=False, ivector_dim=100):
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))
    iterator_scheme = ShuffledScheme(batch_size=batch_size, examples=wsj_dataset.num_examples)
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)

    if use_ivectors:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'ivectors', 'targets'])
        if truncate_ivectors:
            fs = TruncateTransformer(fs, 'ivectors', ivector_dim)
        fs = ConcatenateTransformer(fs, ['features', 'ivectors'], 'features')
    else:
        fs = FilterSources(data_stream=base_stream, sources=['features', 'targets'])
    return Padding(fs)


