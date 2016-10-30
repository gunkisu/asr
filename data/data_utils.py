import hashlib
import numpy as np
import theano
import os
import shutil
from collections import OrderedDict

import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy

from fuel.streams import DataStream
from fuel.transformers import Mapping, ForceFloatX, Padding, SortMapping

from data.schemes import SequentialShuffledScheme
from data.transformers import (MaximumFrameCache, Transpose, Normalize,
                               AddUniformAlignmentMask, WindowFeatures,
                               Reshape, AlignmentPadding, Subsample)


floatX = theano.config.floatX


DATA_PATH = '/data/lisatmp3/speech/'
PHONES_PATH = DATA_PATH + 'timit_processed'

phone_to_phoneme_dict = {'ao':   'aa',
                         'ax':   'ah',
                         'ax-h': 'ah',
                         'axr':  'er',
                         'hv':   'hh',
                         'ix':   'ih',
                         'el':   'l',
                         'em':   'm',
                         'en':   'n',
                         'nx':   'n',
                         'ng':   'eng',
                         'zh':   'sh',
                         'pcl':  'sil',
                         'tcl':  'sil',
                         'kcl':  'sil',
                         'bcl':  'sil',
                         'dcl':  'sil',
                         'gcl':  'sil',
                         'h#':   'sil',
                         'pau':  'sil',
                         'epi':  'sil',
                         'ux':   'uw'}

def sequence_categorical_crossentropy(prediction, targets, mask):
    prediction_flat = prediction.reshape(((prediction.shape[0] *
                                           prediction.shape[1]),
                                          prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    ce = categorical_crossentropy(prediction_flat, targets_flat)
    return T.sum(ce * mask_flat)

def sequence_misclass_rate(prediction, targets, mask):
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    val_targets = targets_flat[mask_flat.nonzero()]
    prediction_reshape = T.neq(T.argmax(prediction, axis=1),
                               val_targets)
    use_length = mask_flat.sum()
    mr = prediction_reshape / T.cast(use_length, 'floatX')
    return T.sum(mr)

def masked_array_to_lists(array, mask):
    return [masked_array_to_list(ar, m) for ar, m in zip(array.T, mask.T)]


def masked_array_to_list(array, mask):
    return [s for s, m in zip(array, mask) if m > 0]


def phone_to_phoneme(phone_mapper, sequence):
    sequence = [s for s in sequence if s not in ('q', '<END>', '<START>')]
    phones = phone_mapper.keys()
    black_list = phone_to_phoneme_dict.keys() + ['q', '<END>', '<START>']
    phones_to_phonemes = dict([(p, p) for p in phones if p not in black_list])
    phones_to_phonemes.update(phone_to_phoneme_dict)
    phonemes = [phones_to_phonemes[s] for s in sequence]
    # remove repetitions
    output = []
    last = 'imunique'
    for p in phonemes:
        if p != last:
            output.append(p)
            last = p
    return output


class Normalizer(object):

    def __init__(self):
        self.sum = 0.0
        self.sum_of_squares = 0.0
        self.N = 0
        self.trained = False

    def fit(self, generator):
        iterator = generator()
        for x in iterator:
            self.sum += x.sum(0)
            self.N += x.shape[0]

        # separate ss pass for numerical stability
        iterator = generator()
        self.x_mean = self.sum / self.N
        for x in iterator:
            self.sum_of_squares += ((x - self.x_mean)**2).sum(0)

        self.x_stdev = np.sqrt(self.sum_of_squares / self.N)
        self.trained = True

    def apply(self, x):
        assert self.trained
        return (x - self.x_mean) / self.x_stdev


def create_mapper(iterator):
    """Returns a dictionary that maps all unique symbols to integers."""
    symbols = set(reduce(lambda x, y: x + y, iterator))
    mapper = dict([(key, value) for value, key in enumerate(list(symbols),
                                                            start=2)])
    mapper['<START>'] = 0
    mapper['<END>'] = 1
    return mapper


def dict_union(*dicts, **kwargs):
    """Return union of a sequence of disjoint dictionaries.

    Parameters
    ----------
    dicts : dicts
        A set of dictionaries with no keys in common. If the first
        dictionary in the sequence is an instance of `OrderedDict`, the
        result will be OrderedDict.
    **kwargs
        Keywords and values to add to the resulting dictionary.

    Raises
    ------
    ValueError
        If a key appears twice in the dictionaries or keyword arguments.

    """
    dicts = list(dicts)
    if dicts and isinstance(dicts[0], OrderedDict):
        result = OrderedDict()
    else:
        result = {}
    for d in list(dicts) + [kwargs]:
        duplicate_keys = set(result.keys()) & set(d.keys())
        if duplicate_keys:
            raise ValueError("The following keys have duplicate entries: {}"
                             .format(", ".join(str(key) for key in
                                               duplicate_keys)))
        result.update(d)
    return result


def shared_zeros(size, name):
    if size == 1:
        return theano.shared(np.cast[floatX](0.), name=name)
    return theano.shared(np.zeros(size, dtype=floatX), name=name)


def shared_uniform(rng, size, low, high, name):
    w = np.asarray(rng.uniform(size, low=low, high=high), dtype=floatX)
    return theano.shared(w, name=name)


def shared_normal(rng, size, stdev, name):
    w = np.asarray(rng.standard_normal(size) * stdev, dtype=floatX)
    return theano.shared(w, name=name)


def apply_mask(non_masked, mask):
    assert len(non_masked) == len(mask)
    masked = non_masked[0:sum(mask) + 1]
    return masked


def key(x):
    return x[0].shape[0]


def construct_stream(dataset, rng, pool_size, maximum_frames, window_features,
                     **kwargs):
    """Construct data stream.

    Parameters:
    -----------
    dataset : Dataset
        Dataset to use.
    rng : numpy.random.RandomState
        Random number generator.
    pool_size : int
        Pool size for TIMIT dataset.
    maximum_frames : int
        Maximum frames for TIMIT datset.
    subsample : bool, optional
        Subsample features.
    pretrain_alignment : bool, optional
        Use phoneme alignment for pretraining.
    uniform_alignment : bool, optional
        Use uniform alignment for pretraining.

    """
    kwargs.setdefault('subsample', False)
    kwargs.setdefault('pretrain_alignment', False)
    kwargs.setdefault('uniform_alignment', False)
    stream = DataStream(
        dataset,
        iteration_scheme=SequentialShuffledScheme(dataset.num_examples,
                                                  pool_size, rng))
    if kwargs['pretrain_alignment'] and kwargs['uniform_alignment']:
        stream = AddUniformAlignmentMask(stream)
    stream = Reshape('features', 'features_shapes', data_stream=stream)
    means, stds = dataset.get_normalization_factors()
    stream = Normalize(stream, means, stds)
    if not window_features == 1:
        stream = WindowFeatures(stream, 'features', window_features)
    if kwargs['pretrain_alignment']:
        stream = Reshape('alignments', 'alignments_shapes', data_stream=stream)
    stream = Mapping(stream,
                     SortMapping(key=key))
    stream = MaximumFrameCache(max_frames=maximum_frames, data_stream=stream,
                               rng=rng)
    stream = Padding(data_stream=stream,
                     mask_sources=['features', 'phonemes'])
    if kwargs['pretrain_alignment']:
        stream = AlignmentPadding(stream, 'alignments')
        stream = Transpose(stream, [(1, 0, 2), (1, 0), (1, 0), (1, 0),
                            (2, 1, 0)])
    else:
        stream = Transpose(stream, [(1, 0, 2), (1, 0), (1, 0), (1, 0)])

    stream = ForceFloatX(stream)
    if kwargs['subsample']:
        stream = Subsample(stream, 'features', 5)
        stream = Subsample(stream, 'features_mask', 5)
    return stream


def file_hash(afile, blocksize=65536):
    buf = afile.read(blocksize)
    hasher = hashlib.md5()
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    return hasher.digest()


def make_local_copy(filename):
    local_name = os.path.join('/Tmp/', os.environ['USER'],
                              os.path.basename(filename))
    if (not os.path.isfile(local_name) or
                file_hash(open(filename)) != file_hash(open(local_name))):
        print '.. made local copy at', local_name
        shutil.copy(filename, local_name)
    return local_name

from blocks.bricks.parallel import Fork