import pytest
import tensorflow as tf
import numpy as np
import itertools
import pandas as pd

from deephub.models.feeders import MemorySamplesFeeder
from deephub.models.feeders.tfrecords.features import ImageFeature


def read_iterator(iterator: tf.data.Iterator):
    """Read all elements of an iterator"""
    results = []
    next_op = iterator.get_next()
    with tf.Session() as session:
        try:
            while True:
                results.append(session.run(next_op))
        except tf.errors.OutOfRangeError:
            pass
    return results


def batch_numpy_array(a, batch_size):
    return np.split(a, np.arange(batch_size, a.shape[0], batch_size))


def equal_list_of_arrays(a, b):
    if a is None or b is None:
        return False  # Sequences are not of the same length

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not a.size == b.size:
            return False  # Arrays are not of the same size
        return np.all(a == b)
    else:
        return all(map(lambda tup: equal_list_of_arrays(tup[0], tup[1]), itertools.zip_longest(a, b, fillvalue=None)))


class TestMemorySamplesFeeder:

    def test_np_with_batch_1(self):
        x = np.arange(0, 10)
        y = np.arange(0, 1).repeat(10)

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=1,
            feed_as_dict=False
        )

        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())
        assert len(results) == 10
        assert results == list(zip(x, y))
        assert feeder.total_steps(1) == 10

    def test_np_with_batch_1_as_dict(self):
        x = np.arange(0, 10)
        y = np.arange(0, 1).repeat(10)

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=1,
            feed_as_dict=True
        )

        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())
        expected = [
            ({0: xi}, {0: yi})
            for xi, yi in zip(x.reshape(-1, 1), y.reshape(-1, 1))
        ]

        assert len(results) == 10
        assert results == expected
        assert feeder.total_steps(1) == 10

    def test_np_with_batch_2(self):
        x = np.arange(0, 10)
        y = np.arange(0, 1).repeat(10)

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=2,
            feed_as_dict=False
        )
        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())
        assert len(results) == 5
        assert equal_list_of_arrays(results, list(zip(batch_numpy_array(x, 2), batch_numpy_array(y, 2))))
        assert feeder.total_steps(1) == 5

    def test_np_with_epochs(self):
        x = np.arange(0, 100)
        y = np.arange(1000, 1100)

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=10,
            feed_as_dict=False
        )

        input_fn = feeder.get_input_fn(5)
        results = read_iterator(input_fn().make_one_shot_iterator())

        assert len(results) == 50
        assert equal_list_of_arrays(
            results,
            list(zip(batch_numpy_array(x, 10),
                     batch_numpy_array(y, 10))
                 ) * 5
        )
        assert feeder.total_steps(5) == 50

    def test_shuffle_with_epochs(self):
        x = np.arange(0, 100)
        y = np.arange(1000, 1100)

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=10,
            shuffle=True,
            feed_as_dict=False
        )

        tf.set_random_seed(1)

        input_fn = feeder.get_input_fn(5)
        results = read_iterator(input_fn().make_one_shot_iterator())

        assert len(results) == 50
        for (batch_x, batch_y) in results:
            assert np.all(batch_x == batch_y - 1000)
        assert feeder.total_steps(5) == 50

    def test_total_steps(self):
        x = np.arange(0, 100)
        y = np.arange(1000, 1100)

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=10,
            shuffle=True,
            feed_as_dict=False
        )

        assert feeder.total_steps(1) == 10
        assert feeder.total_steps(2) == 20

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=7,
            shuffle=True,
            feed_as_dict=False
        )

        assert feeder.total_steps(1) == 15
        assert feeder.total_steps(2) == 29

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=10,
            shuffle=True,
            feed_as_dict=True
        )

        assert feeder.total_steps(1) == 10
        assert feeder.total_steps(2) == 20

    def test_wrong_x_y_shape(self):
        x = np.arange(0, 101)
        y = np.arange(1000, 1100)

        with pytest.raises(ValueError):
            MemorySamplesFeeder(x=x,
                                y=y)

    def test_wrong_x_y_dict_shape(self):
        x = {
            'f1': np.arange(0, 12),
            'f2': np.arange(20, 22),
        }
        y = np.arange(1000, 1100)

        with pytest.raises(ValueError):
            MemorySamplesFeeder(x=x,
                                y=y)

    def test_pandas_input_with_dict(self):
        x = pd.DataFrame(np.arange(0, 20).reshape(-1, 2),
                         columns=['feature_1', 'feature_2'])
        y = pd.DataFrame(np.arange(100, 110))

        feeder = MemorySamplesFeeder(x, y, shuffle=False, batch_size=2, feed_as_dict=True)

        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())

        assert len(results) == 5
        assert np.all(results[0][0]['feature_1'] == np.array([0, 2]))
        assert np.all(results[0][0]['feature_2'] == np.array([1, 3]))
        assert np.all(results[0][1][0] == np.array([100, 101]))

    def test_dict_input_with_1d(self):
        x = {
            'feature1': [1, 2, 3, 4]
        }
        y = {
            'label': [4, 5, 6, 7]
        }

        feeder = MemorySamplesFeeder(x, y, shuffle=False, batch_size=2, feed_as_dict=True)
        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())

        assert len(results) == 2
        assert np.all(results[0][0]['feature1'] == np.array([1, 2]))
        assert np.all(results[0][1]['label'] == np.array([4, 5]))

    def test_dict_input_with_2d(self):
        x = {
            'feature1': [[1, 11], [2, 12], [3, 13], [4, 14]]
        }
        y = {
            'label': [4, 5, 6, 7]
        }

        feeder = MemorySamplesFeeder(x, y, shuffle=False, batch_size=2, feed_as_dict=True)
        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())
        assert len(results) == 2
        assert np.all(results[0][0]['feature1'] == np.array([[1, 11], [2, 12]]))
        assert np.all(results[0][1]['label'] == np.array([4, 5]))

    def test_python_single_list(self):
        x = list(range(20))
        y = list(range(100, 120))

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            shuffle=False, batch_size=2,
            feed_as_dict=False)

        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())

        expected = list(zip(batch_numpy_array(np.arange(20), 2),
                            batch_numpy_array(np.arange(100, 120), 2)))
        assert equal_list_of_arrays(results, expected)

    def test_python_nested_list(self):
        x = list(zip(range(20), range(100, 120)))
        y = list(range(1000, 1020))

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            shuffle=False, batch_size=2,
            feed_as_dict=False)

        input_fn = feeder.get_input_fn(1)
        results = read_iterator(input_fn().make_one_shot_iterator())

        expected = list(zip(batch_numpy_array(np.hstack((np.arange(20).reshape(-1, 1),
                                                         np.arange(100, 120).reshape(-1, 1))), 2),
                            batch_numpy_array(np.arange(1000, 1020), 2)))
        assert equal_list_of_arrays(results, expected)

    def test_repr(self):
        x = np.arange(0, 100)
        y = np.arange(1000, 1100)

        res = repr(MemorySamplesFeeder(x=x, y=y, batch_size=16))
        assert res == "MemorySamplesFeeder(x=(100, 1), y=(100, 1), batch_size=16, " \
                      "shuffle=False, as_dict=True)"


class TestImageFeatureType:

    def _parse_image(self, img_feature, path):
        with open(path, 'rb') as f:
            content = tf.constant(f.read(), dtype=tf.string)

        with tf.Session() as s:
            return s.run(img_feature.post_parse_op(content))

    def test_op_dimensions_no_resize(self):
        img_feature = ImageFeature(decode_format='jpeg')
        op = img_feature.post_parse_op(tf.placeholder(tf.string))

        assert op.shape.as_list() == [None, None, 3]

    def test_op_dimensions_with_resize(self):
        img_feature = ImageFeature(128, 135, decode_format='jpeg')
        op = img_feature.post_parse_op(tf.placeholder(tf.string))

        assert op.shape.as_list() == [128, 135, 3]

    def test_op_dimensions_with_resize_auto(self):
        img_feature = ImageFeature(128, 135, decode_format='auto')
        op = img_feature.post_parse_op(tf.placeholder(tf.string))

        assert op.shape.as_list() == [128, 135, None]

    def test_op_dimensions_no_resize_gray(self):
        img_feature = ImageFeature(128, 135, decode_format='jpeg', channels=ImageFeature.CHANNELS_GRAYSCALE)
        op = img_feature.post_parse_op(tf.placeholder(tf.string))

        assert op.shape.as_list() == [128, 135, 1]

    def test_decode_jpeg(self, datadir):
        img_feature = ImageFeature(decode_format='jpeg')

        # Check decoding
        decoded = self._parse_image(img_feature, datadir / 'images' / 'lena.jpg')
        assert decoded.shape == (256, 256, 3)

    def test_decode_jpeg_and_resize_and_gray_scale(self, datadir):
        img_feature = ImageFeature(122, 300, decode_format='jpeg', channels=ImageFeature.CHANNELS_GRAYSCALE)

        # Check decoding
        decoded = self._parse_image(img_feature, datadir / 'images' / 'lena.jpg')
        assert decoded.shape == (122, 300, 1)

    def test_unknown_decoder(self):
        with pytest.raises(ValueError):
            ImageFeature(decode_format='unknown')

    def test_partial_resize(self):
        with pytest.raises(ValueError):
            ImageFeature(resize_height=160, decode_format='jpeg')
