from pathlib import Path
import pytest
import tensorflow as tf
import numpy as np
import itertools
import pandas as pd
import os

from deephub.models.feeders import MemorySamplesFeeder, TFRecordExamplesFeeder


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


class TestTfRecordExamplesFeeder:
    FILE_PATH = '../../../runtime_resources/data/yahoo_answers/legacy_setup/train_char.tfr'
    DATASET_TOTAL_EXAMPLES = 8
    NUM_EXAMPLES = [1, 4, 8]
    BATCH_SIZE = [1, 2, 6, 8]
    EPOCHS = [1, 2, 5]
    SHUFFLE = True

    @staticmethod
    def _get_feeder(path, max_examples: int, batch_size: int, shuffle: bool = False, drop_remainder: bool = False) -> TFRecordExamplesFeeder:
        tf_record_path = path / 'tfrecords' / 'text' / 'train_char.tfr'

        return TFRecordExamplesFeeder(file_patterns=tf_record_path,
                                      max_examples=max_examples,
                                      features_map={'text': tf.io.FixedLenFeature(1024, tf.int64)},
                                      labels_map={'label': tf.io.FixedLenFeature(1, tf.int64)},
                                      batch_size=batch_size,
                                      drop_remainder=drop_remainder,
                                      shuffle=shuffle,
                                      num_parallel_maps=None,
                                      num_parallel_reads=None,
                                      prefetch=None)

    def test_number_of_epochs(self, datadir):
        """
        Call feeder with batch size equal to number of examples, and check the size of the results which is
        the total number of epochs, as 1 step == 1 epoch.
        :return: None
        """
        # Outer dict for NUM_EXAMPLES, Inner dict for EPOCHS
        GROUND_TRUTH_RESULT = {'1': {'1': 1, '2': 2, '5': 5},
                               '4': {'1': 1, '2': 2, '5': 5},
                               '8': {'1': 1, '2': 2, '5': 5}
                               }

        for val1 in TestTfRecordExamplesFeeder.NUM_EXAMPLES:
            for val2 in TestTfRecordExamplesFeeder.EPOCHS:
                feeder = TestTfRecordExamplesFeeder._get_feeder(path=datadir,
                                                                max_examples=val1,
                                                                batch_size=val1,
                                                                shuffle=TestTfRecordExamplesFeeder.SHUFFLE)
                input_fn = feeder.get_input_fn(epochs=val2)
                results = read_iterator(input_fn().make_one_shot_iterator())

                assert len(results) == GROUND_TRUTH_RESULT[str(val1)][str(val2)], \
                    "{} != {}, val1: {}, val2: {}".format(len(results),
                                                          GROUND_TRUTH_RESULT[str(val1)][str(val2)],
                                                          val1, val2)

    def test_number_of_steps(self, datadir):
        """
        Call feeder with all the available examples, with different batch sizes and check the final
        number of steps.
        :return: None
        """
        # Outer dict for BATCH_SIZE, Inner dict for EPOCHS
        GROUND_TRUTH_RESULT = {'1': {'1': 8, '2': 16, '5': 40},
                               '2': {'1': 4, '2': 8, '5': 20},
                               '6': {'1': 2, '2': 3, '5': 7},
                               '8': {'1': 1, '2': 2, '5': 5}
                               }

        for val1 in TestTfRecordExamplesFeeder.BATCH_SIZE:
            for val2 in TestTfRecordExamplesFeeder.EPOCHS:
                feeder = TestTfRecordExamplesFeeder._get_feeder(
                    path=datadir,
                    max_examples=-1,
                    batch_size=val1,
                    shuffle=TestTfRecordExamplesFeeder.SHUFFLE)
                input_fn = feeder.get_input_fn(epochs=val2)
                results = read_iterator(input_fn().make_one_shot_iterator())

                steps = feeder.total_steps(epochs=val2)
                assert len(results) == steps == GROUND_TRUTH_RESULT[str(val1)][str(val2)], \
                    "{} != {} != {}, val1: {}, val2:{}".format(len(results), steps,
                                                               GROUND_TRUTH_RESULT[str(val1)][str(val2)],
                                                               val1, val2)

    def test_number_of_examples(self, datadir):
        """
        Call feeder with different number of examples, different batch sizes and different number of epochs,
        and check that the dimensionality of the returned batch examples matches with the ground truths.
        :return: None
        """
        # Outer dict for NUM_EXAMPLES. Next dict for BATCH_SIZE. Inner dict for EPOCHS.
        GROUND_TRUTH_RESULT = {'1': {'1': {'1': [1],
                                           '2': [1, 1],
                                           '5': [1 for _ in range(5)]
                                           },
                                     '2': {'1': [1],
                                           '2': [1, 1],
                                           '5': [1 for _ in range(5)]
                                           },
                                     '6': {'1': [1],
                                           '2': [1, 1],
                                           '5': [1 for _ in range(5)]
                                           },
                                     '8': {'1': [1],
                                           '2': [1, 1],
                                           '5': [1 for _ in range(5)]
                                           }
                                     },
                               '4': {'1': {'1': [1 for _ in range(4)],
                                           '2': [1 for _ in range(8)],
                                           '5': [1 for _ in range(20)]
                                           },
                                     '2': {'1': [2, 2],
                                           '2': [2 for _ in range(4)],
                                           '5': [2 for _ in range(10)]
                                           },
                                     '6': {'1': [4],
                                           '2': [4, 4],
                                           '5': [4 for _ in range(5)]
                                           },
                                     '8': {'1': [4],
                                           '2': [4, 4],
                                           '5': [4 for _ in range(5)]
                                           }
                                     },
                               '8': {'1': {'1': [1 for _ in range(8)],
                                           '2': [1 for _ in range(16)],
                                           '5': [1 for _ in range(40)]
                                           },
                                     '2': {'1': [2 for _ in range(4)],
                                           '2': [2 for _ in range(8)],
                                           '5': [2 for _ in range(20)]
                                           },
                                     '6': {'1': [6, 2],
                                           '2': [6, 6, 4],
                                           '5': [6, 6, 6, 6, 6, 6, 4]
                                           },
                                     '8': {'1': [8],
                                           '2': [8, 8],
                                           '5': [8 for _ in range(5)]
                                           }
                                     }
                               }

        for val1 in TestTfRecordExamplesFeeder.NUM_EXAMPLES:
            for val2 in TestTfRecordExamplesFeeder.BATCH_SIZE:
                for val3 in TestTfRecordExamplesFeeder.EPOCHS:
                    if val2 > val1:
                        with pytest.raises(ValueError):  # batch_size > max_examples
                            TestTfRecordExamplesFeeder._get_feeder(path=datadir,
                                                                   max_examples=val1,
                                                                   batch_size=val2,
                                                                   shuffle=TestTfRecordExamplesFeeder.SHUFFLE)
                        continue
                    else:
                        feeder = TestTfRecordExamplesFeeder._get_feeder(path=datadir,
                                                                        max_examples=val1,
                                                                        batch_size=val2,
                                                                        shuffle=TestTfRecordExamplesFeeder.SHUFFLE)

                    input_fn = feeder.get_input_fn(epochs=val3)
                    results = read_iterator(input_fn().make_one_shot_iterator())

                    index = 0
                    for res in results:
                        assert res[0]['text'].shape[0] == GROUND_TRUTH_RESULT[str(val1)][str(val2)][str(val3)][index], \
                            "{} != {}, val1: {}, val2: {}, val3: {}".format(res[0]['text'].shape[0],
                                                                            GROUND_TRUTH_RESULT[str(val1)][str(val2)][
                                                                                str(val3)][index],
                                                                            val1, val2, val3)
                        assert res[1]['label'].shape[0] == GROUND_TRUTH_RESULT[str(val1)][str(val2)][str(val3)][index], \
                            "{} != {}, val1: {}, val2: {}, val3: {}".format(res[1]['label'].shape[0],
                                                                            GROUND_TRUTH_RESULT[str(val1)][str(val2)][
                                                                                str(val3)][index],
                                                                            val1, val2, val3)
                        index += 1

    def test_data_batching(self, datadir):
        # Read all data at once, for 1 epoch
        feeder = TestTfRecordExamplesFeeder._get_feeder(
            path=datadir,
            max_examples=-1,
            batch_size=8,
            shuffle=False)
        input_fn = feeder.get_input_fn(epochs=1)
        ground_truth_results = read_iterator(input_fn().make_one_shot_iterator())

        for val1 in TestTfRecordExamplesFeeder.BATCH_SIZE:
            for val2 in TestTfRecordExamplesFeeder.EPOCHS:
                feeder = TestTfRecordExamplesFeeder._get_feeder(path=datadir,
                                                                max_examples=-1,
                                                                batch_size=val1,
                                                                shuffle=False)
                input_fn = feeder.get_input_fn(epochs=val2)
                results = read_iterator(input_fn().make_one_shot_iterator())

                index = 0
                for res in results:
                    for bs_index in range(res[0]['text'].shape[0]):
                        np.testing.assert_array_equal(ground_truth_results[0][0]['text'][index],
                                                      res[0]['text'][bs_index])

                        np.testing.assert_array_equal(ground_truth_results[0][1]['label'][index],
                                                      res[1]['label'][bs_index])

                        index += 1
                        index = index % 8

    def test_drop_remainder(self, datadir):
        # Total examples are 8
        feeder = TestTfRecordExamplesFeeder._get_feeder(
            path=datadir,
            max_examples=-1,
            batch_size=3,
            drop_remainder=True,
            shuffle=False)
        # 8 total_examples with batch_size=3, drop total_examples % batch_size, steps equal to 2
        assert feeder.total_steps(epochs=1) == 2

        feeder = TestTfRecordExamplesFeeder._get_feeder(
            path=datadir,
            max_examples=-1,
            batch_size=4,
            drop_remainder=True,
            shuffle=False)
        # 8 total_examples with batch_size=4, drop total_examples % batch_size, steps equal to 2
        assert feeder.total_steps(epochs=1) == 2

        feeder = TestTfRecordExamplesFeeder._get_feeder(
            path=datadir,
            max_examples=-1,
            batch_size=3,
            drop_remainder=False,
            shuffle=False)
        # 8 total_examples with batch_size=3, NOT drop total_examples % batch_size, steps equal to 3
        assert feeder.total_steps(epochs=1) == 3


class TestMemorySamplesFeeder:

    def test_np_with_batch_1(self):
        x = np.arange(0, 10)
        y = np.arange(0, 1).repeat(10)

        feeder = MemorySamplesFeeder(
            x=x,
            y=y,
            batch_size=1,
            feed_as_dict=False,
            shuffle=False
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
            feed_as_dict=True,
            shuffle=False
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
            feed_as_dict=False,
            shuffle=False
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
            feed_as_dict=False,
            shuffle=False
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

        res = repr(MemorySamplesFeeder(x=x, y=y, batch_size=16, shuffle=False))
        assert res == "MemorySamplesFeeder(x=(100, 1), y=(100, 1), batch_size=16, " \
                      "shuffle=False, as_dict=True)"
