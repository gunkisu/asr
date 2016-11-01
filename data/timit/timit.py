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
class FramewiseTimit(PytablesDataset):
    def __init__(self,
                 which_set='train',
                 path=None,
                 local_copy=True):
        if path:
            self.path = path
        else:
            self.path = '/data/lisatmp3/speech/timit_fbank_framewise.h5'
        if local_copy and not self.path.startswith('/Tmp'):
            self.path = make_local_copy(self.path)
        self.which_set = which_set
        self.sources = ('features', 'features_shapes', 'phonemes')
        super(FramewiseTimit, self).__init__(self.path,
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

####################
# non-aligned data #
####################
class Timit(PytablesDataset):
    """TIMIT dataset.

    Parameters
    ----------
    which_set : str, opt
        either 'train', 'dev' or 'test'.
    alignment : bool
        Whether return alignment.
    features : str
        The features to use. They will lead to the correct h5 file.
    """
    def __init__(self,
                 path,
                 which_set='train',
                 alignment=False,
                 local_copy=True):
        self.path=path
        if local_copy and not self.path.startswith('/Tmp'):
            self.path = make_local_copy(self.path)
        self.which_set = which_set
        if alignment:
            self.sources = ('features', 'features_shapes', 'phonemes',
                            'alignments', 'alignments_shapes')
        else:
            self.sources = ('features', 'features_shapes', 'phonemes')
        super(Timit, self).__init__(
            self.path, self.sources, data_node=which_set)

    def get_phoneme_dict(self):
        phoneme_list = self.h5file.root._v_attrs.phones_list
        return OrderedDict(enumerate(phoneme_list))

    def get_phoneme_ind_dict(self):
        phoneme_list = self.h5file.root._v_attrs.phones_list
        return OrderedDict(zip(phoneme_list, range(len(phoneme_list))))

    def get_normalization_factors(self):
        means = self.h5file.root._v_attrs.means
        stds = self.h5file.root._v_attrs.stds
        return means, stds

    def open_file(self, path):
        # CAUTION: This is a hack!
        # Use `open_file` when Fred updates os
        self.h5file = tables.File(path, mode="r")
        node = self.h5file.get_node('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]
        if self.stop is None:
            self.stop = self.nodes[0].nrows
        self.num_examples = self.stop - self.start

def framewise_timit_datastream(path,
                               which_set,
                               batch_size,
                               local_copy=False):
    # load frame-wise dataset
    timit_dataset = FramewiseTimit(which_set=which_set,
                                   path=path,
                                   local_copy=local_copy)

    # set shuffle range
    shuffle_rng = numpy.random.RandomState(123)

    # set iterator scheme
    iterator_scheme = SequentialShuffledScheme(num_examples=timit_dataset.num_examples,
                                               batch_size=batch_size,
                                               rng=shuffle_rng)

    # base data stream
    base_stream = DataStream(dataset=timit_dataset,
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

def timit_datastream(path,
                     which_set,
                     local_copy,
                     pool_size,
                     maximum_frames):

    # load dataset
    timit_dataset = Timit(which_set=which_set,
                          path=path,
                          local_copy=local_copy)

    # get statistics
    data_means, data_stds = timit_dataset.get_normalization_factors()

    # set shuffle range
    shuffle_rng = numpy.random.RandomState(123)

    # set iterator scheme
    iterator_scheme = SequentialShuffledScheme(num_examples=timit_dataset.num_examples,
                                               batch_size=pool_size,
                                               rng=shuffle_rng)

    # base data stream
    base_stream = DataStream(dataset=timit_dataset,
                             iteration_scheme=iterator_scheme)

    # reshape stream
    reshape_stream = Reshape(data_source='features',
                             shape_source='features_shapes',
                             data_stream=base_stream)

    # normalize data stream
    normalize_stream = Normalize(data_stream=reshape_stream,
                                 means=data_means,
                                 stds=data_stds)

    # sort data stream
    sort_stream = Mapping(data_stream=normalize_stream,
                          mapping=SortMapping(key=lambda x: x[0].shape[0]))

    # max frame stream
    max_frame_stream = MaximumFrameCache(max_frames=maximum_frames,
                                         data_stream=sort_stream,
                                         rng=shuffle_rng)

    # padding data stream
    padded_stream = Padding(data_stream=max_frame_stream,
                            mask_sources=['features', 'phonemes'])

    # floatX stream
    data_stream = ForceFloatX(padded_stream)
    return timit_dataset,  data_stream
