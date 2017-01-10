from collections import OrderedDict

import numpy

from theano import config
import itertools

from fuel.transformers import Transformer
from picklable_itertools.extras import equizip

from six import iteritems

class ConcatenateTransformer(Transformer):
    '''Concatenate data sources into one data source.
    The order of the data sources will be the same'''
    def __init__(self, data_stream, concat_sources, new_source=None, **kwargs):
        if data_stream.produces_examples:
                 raise ValueError('the wrapped data stream must produce batches of '
                                                      'examples, not examples')
        if any(source not in data_stream.sources for source in concat_sources):
            raise ValueError("sources must all be contained in "
                             "data_stream.sources")

        self.new_source = new_source if new_source else '_'.join(concat_sources)
        if data_stream.axis_labels:
            axis_labels = dict((source, labels) for (source, labels)
                    in iteritems(data_stream.axis_labels)
                        if source not in concat_sources) 
            axis_labels[self.new_source] = 'concatenated source: {}'.format(concat_sources)
            kwargs.setdefault('axis_labels', axis_labels)
        
        super(ConcatenateTransformer, self).__init__(
            data_stream=data_stream,
            produces_examples=False,
            **kwargs)

        insert_pos = self.data_stream.sources.index(concat_sources[0])
        new_sources = [s for s in data_stream.sources if s not in concat_sources]
        new_sources.insert(insert_pos, self.new_source)
        self.sources = tuple(new_sources)
        self.concat_sources = concat_sources
       
    def transform_batch(self, batch):
        trans_data = []
        src_indices = [self.data_stream.sources.index(s) for s in self.concat_sources]
        data_from_concat_sources = [batch[i] for i in src_indices]
        for examples in itertools.izip(*data_from_concat_sources):
            trans_data.append(numpy.concatenate(examples, axis=1))
        insert_pos = self.data_stream.sources.index(self.concat_sources[0])
        batch = [d for i, d in enumerate(batch) if i not in src_indices]
        batch.insert(insert_pos, trans_data)
        return numpy.asarray(batch)

class TruncateTransformer(Transformer):
    '''Truncate the dimension of a specific data source from the begining'''
    def __init__(self, data_stream, target_source, dim, **kwargs):
        if data_stream.produces_examples:
                 raise ValueError('the wrapped data stream must produce batches of '
                                                      'examples, not examples')
        if target_source not in data_stream.sources:
            raise ValueError("source must be contained in "
                             "data_stream.sources")

        super(TruncateTransformer, self).__init__(
            data_stream=data_stream,
            produces_examples=False,
            **kwargs)

        self.target_source = target_source
        self.dim = dim
       
    def transform_batch(self, batch):
        trans_data = []
        src_idx = self.data_stream.sources.index(self.target_source)
        for ex in batch[src_idx]:
            trans_data.append(ex[:,:self.dim])
        batch[src_idx] = trans_data
        return numpy.asarray(batch)


class MaximumFrameCache(Transformer):
    """Cache examples, and create batches of maximum number of frames.

    Given a data stream which reads large chunks of data, this data
    stream caches these chunks and returns batches with a maximum number
    of acoustic frames.

    Parameters
    ----------
    max_frames : int
        maximum number of frames per batch

    Attributes
    ----------
    cache : list of lists of objects
        This attribute holds the cache at any given point. It is a list of
        the same size as the :attr:`sources` attribute. Each element in
        this list is a deque of examples that are currently in the
        cache. The cache gets emptied at the start of each epoch, and gets
        refilled when needed through the :meth:`get_data` method.

    """

    def __init__(self, data_stream, max_frames, rng):
        super(MaximumFrameCache, self).__init__(
            data_stream)
        self.max_frames = max_frames
        self.cache = OrderedDict([(name, []) for name in self.sources])
        self.num_frames = []
        self.rng = rng
        self.produces_examples = False

    def next_request(self):
        curr_max = 0
        for i, n_frames in enumerate(self.num_frames):
            # Select max number of frames because of future padding
            curr_max = max(n_frames, curr_max)
            total = curr_max * (i + 1)
            if total >= self.max_frames:
                return i + 1
        return len(self.num_frames)

    def get_data(self, request=None):
        if not self.cache[self.cache.keys()[0]]:
            self._cache()
        data = []
        request = self.next_request()
        for source_name in self.cache:
            data.append(numpy.asarray(self.cache[source_name][:request]))
        self.cache = OrderedDict([(name, dt[request:]) for name, dt
                                  in self.cache.iteritems()])
        self.num_frames = self.num_frames[request:]

        return tuple(data)

    def get_epoch_iterator(self, **kwargs):
        self.cache = OrderedDict([(name, []) for name in self.sources])
        self.num_frames = []
        return super(MaximumFrameCache, self).get_epoch_iterator(**kwargs)

    def _cache(self):
        data = next(self.child_epoch_iterator)
        indexes = range(len(data[0]))
        # self.rng.shuffle(indexes)
        data = [[dt[i] for i in indexes] for dt in data]
        self.cache = OrderedDict([(name, self.cache[name] + dt) for name, dt
                                  in equizip(self.data_stream.sources, data)])
        self.num_frames.extend([x.shape[0] for x in data[0]])


class Transpose(Transformer):
    """Transpose axes of datastream.
    """

    def __init__(self, datastream, axes_list):
        super(Transpose, self).__init__(datastream)
        self.axes_list = axes_list
        self.produces_examples = False

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transposed_data = []
        for axes, data in zip(self.axes_list, data):
            transposed_data.append(numpy.transpose(data, axes))
        return transposed_data


class AddUniformAlignmentMask(Transformer):
    """Adds an uniform alignment mask to the incoming batch.

    Parameters
    ----------

    """

    def __init__(self, data_stream):
        super(AddUniformAlignmentMask, self).__init__(data_stream)
        self.sources = self.data_stream.sources + ('alignment',)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        sources = self.data_stream.sources

        x_idx = sources.index('x')
        y_idx = sources.index('y')
        x_mask_idx = sources.index('x_mask')
        y_mask_idx = sources.index('y_mask')

        batch_size = data[x_idx].shape[1]
        max_len_output = data[y_idx].shape[0]
        max_len_input = data[x_idx].shape[0]
        mask_shape = (max_len_output, batch_size, max_len_input)
        alignment = numpy.zeros(mask_shape, dtype=config.floatX)

        for k in xrange(batch_size):
            in_size = numpy.count_nonzero(data[x_mask_idx][:, k])
            out_size = numpy.count_nonzero(data[y_mask_idx][:, k])
            n = int(in_size / out_size)  # Maybe clever way than int to do this
            v = numpy.hstack([numpy.ones(n, dtype=config.floatX),
                              numpy.zeros(max_len_input - n,
                                          dtype=config.floatX)])
            alignment[0, k] = v
            for i in xrange(1, out_size):
                alignment[i, k] = numpy.roll(v, i * n)

                # DEBUG
                # plt.figure()
                # plt.imshow(alignment[:,k,:], cmap='gray', interpolation='none')
                # plt.show()
        data = data + (alignment,)

        return data


class AlignmentPadding(Transformer):
    def __init__(self, data_stream, alignment_source):
        super(AlignmentPadding, self).__init__(data_stream)
        self.alignment_source = alignment_source

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(equizip(self.sources, data))

        alignments = data[self.alignment_source]

        input_lengths = [alignment.shape[1] for alignment in alignments]
        output_lengths = [alignment.shape[0] for alignment in alignments]
        max_input_length = max(input_lengths)
        max_output_length = max(output_lengths)

        batch_size = len(alignments)

        padded_alignments = numpy.zeros((max_output_length, batch_size,
                                         max_input_length))

        for i, alignment in enumerate(alignments):
            out_size, inp_size = alignment.shape
            padded_alignments[:out_size, i, :inp_size] = alignment

        data[self.alignment_source] = padded_alignments

        return data.values()


class Reshape(Transformer):
    """Reshapes data in the stream according to shape source."""

    def __init__(self, data_source, shape_source, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.data_source = data_source
        self.shape_source = shape_source
        self.sources = tuple(source for source in self.data_stream.sources
                             if source != shape_source)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        shapes = data.pop(self.shape_source)
        reshaped_data = []
        for dt, shape in zip(data[self.data_source], shapes):
            reshaped_data.append(dt.reshape(shape))
        data[self.data_source] = reshaped_data
        return data.values()


class Subsample(Transformer):
    def __init__(self, data_stream, source, step):
        super(Subsample, self).__init__(data_stream)
        self.source = source
        self.step = step

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(equizip(self.sources, data))
        dt = data[self.source]

        indexes = ((slice(None, None, self.step),) +
                   (slice(None),) * (len(dt.shape) - 1))
        subsampled = dt[indexes]
        data[self.source] = subsampled
        return data.values()


class WindowFeatures(Transformer):
    def __init__(self, data_stream, source, window_size):
        super(WindowFeatures, self).__init__(data_stream)
        self.source = source
        self.window_size = window_size

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(equizip(self.sources, data))
        feature_batch = data[self.source]

        windowed_features = []
        for features in feature_batch:
            features_padded = features.copy()

            features_shifted = [features]
            # shift forward
            for i in xrange(self.window_size / 2):
                feats = numpy.roll(features_padded, i + 1, axis=0)
                feats[:i + 1, :] = 0
                features_shifted.append(feats)
            features_padded = features.copy()

            # shift backward
            for i in xrange(self.window_size / 2):
                feats = numpy.roll(features_padded, -i - 1, axis=0)
                feats[-i - 1:, :] = 0
                features_shifted.append(numpy.roll(features_padded, -i - 1,
                                                   axis=0))
            windowed_features.append(numpy.concatenate(
                features_shifted, axis=1))
        data[self.source] = windowed_features
        return data.values()


class Normalize(Transformer):
    """Normalizes each features : x = (x - means)/stds"""

    def __init__(self, data_stream, means, stds):
        super(Normalize, self).__init__(data_stream)
        self.means = means
        self.stds = stds

        self.produces_examples = False

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        for i in range(len(data['features'])):
            data['features'][i] -= self.means
            data['features'][i] /= self.stds
        return data.values()

def length_getter(dt):
    def get_length(k):
        return dt[k].shape[0]

    return get_length

class SortByLegth(Transformer):
    def __init__(self, data_stream, source='features'):
        super(SortByLegth, self).__init__(data_stream)
        self.source = source

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        dt = data[self.source]
        indexes = sorted(range(len(dt)), key=length_getter(dt))
        for source in self.sources:
            data[source] = [data[source][k] for k in indexes]
        return data.values()
