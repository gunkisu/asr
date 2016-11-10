import numpy
import tables
from collections import OrderedDict
from fuel.datasets.hdf5 import PytablesDataset
from data.data_utils import file_hash, make_local_copy
from fuel.streams import DataStream
from fuel.transformers import (Mapping, Padding, SortMapping, ForceFloatX)
from data.transformers import Reshape, Normalize, MaximumFrameCache
from data.schemes import SequentialShuffledScheme

###############
# frame level #
###############
class FramewiseWSJ(PytablesDataset):
    def __init__(self,
                 which_set='train_si284_tr90',
                 path=None,
                 local_copy=True):
        if path:
            self.path = path
        else:
            self.path = '/data/lisatmp3/speech/timit_fbank_framewise.h5'
        if local_copy and not self.path.startswith('/Tmp'):
            self.path = make_local_copy(self.path)
        self.which_set = which_set
        self.sources = ('features', 'features_shapes', 'targets')
        super(FramewiseWSJ, self).__init__(self.path,
                                             self.sources,
                                             data_node=which_set)

    def open_file(self, path):
        # CAUTION: This is a hack!
        # Use `open_file` when Fred updates os
        self.h5file = tables.File(path, mode="r")
        node = self.h5file.get_node('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]
        if self.stop is None:
            self.stop = self.nodes[0].nrows
        self.num_examples = self.stop - self.start


def framewise_wsj_datastream(path,
                               which_set,
                               batch_size,
                               local_copy=False):
    # load frame-wise dataset
    wsj_dataset = FramewiseWSJ(which_set=which_set,
                                   path=path,
                                   local_copy=local_copy)

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
    reshape_stream = Reshape(data_source='features',
                             shape_source='features_shapes',
                             data_stream=base_stream,
                             iteration_scheme=iterator_scheme)

    # sort data stream
    sort_stream = Mapping(data_stream=reshape_stream,
                          mapping=SortMapping(key=lambda x: x[0].shape[0]))

    # padding data stream
    padded_stream = Padding(data_stream=sort_stream)

    return padded_stream

