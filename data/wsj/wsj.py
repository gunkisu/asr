import numpy
import tables
from collections import OrderedDict
from fuel.datasets.hdf5 import H5PYDataset
from data.data_utils import file_hash, make_local_copy
from fuel.streams import DataStream
from fuel.transformers import (Mapping, Padding, SortMapping, ForceFloatX)
from data.transformers import Normalize, MaximumFrameCache
from data.schemes import SequentialShuffledScheme

def framewise_wsj_datastream(path,
                               which_set,
                               batch_size,
                               local_copy=False):
    
    # load frame-wise dataset
    wsj_dataset = H5PYDataset(path, which_sets=(which_set, ))

    # set shuffle range
    shuffle_rng = numpy.random.RandomState(123)

    # set iterator scheme
    iterator_scheme = SequentialShuffledScheme(num_examples=wsj_dataset.num_examples,
                                               batch_size=batch_size,
                                               rng=shuffle_rng)

    # base data stream
    base_stream = DataStream(dataset=wsj_dataset,
                             iteration_scheme=iterator_scheme)

    # reshape data stream data_source, shape_source
#   reshape_stream = Reshape(data_source='fmllr_feat',
#                             shape_source='fmllr_feat_shapes',
#                             data_stream=base_stream,
#                             iteration_scheme=iterator_scheme)

    # sort data stream
#    sort_stream = Mapping(data_stream=base_stream,
#                          mapping=SortMapping(key=lambda x: x[0].shape[0]))

    # padding data stream
    padded_stream = Padding(data_stream=base_stream)

    return padded_stream

