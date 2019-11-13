import logging
from abc import abstractmethod
from typing import Union, Optional, List, Dict, Any

import tensorflow as tf
import numpy as np
import pandas as pd

from .base import FeederBase, InputFN

DataFrameCompat = Union[List[Dict], Dict[str, List], Dict[str, np.ndarray]]
NpOrDataFrame = Union[np.ndarray, pd.DataFrame]

logger = logging.getLogger(__name__)


class MemorySamplesFeeder(FeederBase):
    """
    Feeder from in-memory stored samples.

    This feeder can be used to feed a model training or evaluation data from in-memory data structures like
    numpy or pandas DataFrames. Typically the features are expected to be mapped by their name in a dictionary,
    If it is not the case it will try to make the conversion in a dictionary either from pandas columns names
    or from array column indices. The data can be optionally shuffled and will be returned in batches
    according to a batch size.


    Example to create a feeder from a DataFrame (this can only work for scalar feature columns)

    >>> x = pd.DataFrame(np.arange(0, 100).reshape(-1, 2), columns=['feature1', 'feature2'])
    >>> y = np.arange(1000, 1100)
    >>> feeder = MemorySamplesFeeder(x, y, batch_size=2, feed_as_dict=True)
    >>> # Expected batch format: features = { 'feature1': pd.Series[...], 'feature2': pd.Series[...]}
    >>> #                        labels = {0: np.array()}

    Example to create a feeder from a dictionary of non-scalar features

    >>> x = { '2d': np.arange(0, 10*32*32).reshape(-1, 32, 32)}
    >>> y = np.arange(1000, 1100)
    >>> feeder = MemorySamplesFeeder(x, y, batch_size=2, feed_as_dict=True)
    >>> # Expected batch format: features = { 'feature1': np.array([[[..32*32]], [[..32*32]] ]}
    >>> #                        labels = {0: np.array()}

    Example to create a feeder from numpy array with mini-batch size 2 that does not return dictionary format:

    >>> x = np.arange(0, 10)
    >>> y = np.arange(100, 110)
    >>> feeder = MemorySamplesFeeder(x, y, batch_size=2, feed_as_dict=False)
    >>> # Expected batch format: features = np.array([0,2]),
    >>> #                        labels = np.array(100, 101)

    """

    def __init__(self,
                 x: Union[NpOrDataFrame, DataFrameCompat, Dict[str, NpOrDataFrame]],
                 y: Union[NpOrDataFrame, DataFrameCompat, Dict[str, NpOrDataFrame]],
                 feed_as_dict: bool = True,
                 batch_size: int = 128,
                 drop_remainder: bool = False,
                 shuffle: bool = True):
        """
        Initialize in-memory feeder
        :param x: The x value that will be used for features in the model. It can be a dictionary of columns
            with values for each example. For scalar features, a dataframe can be used to directly map feature
            columns. If a python list or numpy array is given, features will be mapped by their column index.
        :param y: The y value that will be used for labels in model. It follows the same rules as 'x' argument
        :param feed_as_dict: A flag to returns batches in columns batched by their feature name for both features
            and labels of model_fn. If DataFrames are given then the names are selected from their column
            names, otherwise it is the index of the column.
        :param batch_size: The size of mini-batches
        :param drop_remainder: Determines if you drop or not the remainder of data rows (total examples%batch_size)
        :param shuffle: Control if samples should be shuffled before batching.
        """

        any_input_dict = isinstance(x, dict) or isinstance(y, dict)

        self.original_x_shape = None
        self.original_y_shape = None
        self.feed_as_dict = feed_as_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder

        if any_input_dict and not feed_as_dict:
            raise ValueError("You cannot provide X or Y in dict format with 'feed_as_dict=False'")

        # Initialize variables
        self.x = self._initialized_input(x)
        self.y = self._initialized_input(y)

        # Extract number of examples
        if not feed_as_dict:
            self.original_x_shape = self.x.shape
            self.original_y_shape = self.y.shape
        else:
            self.original_x_shape = self._get_shape_from_dict(self.x)
            self.original_y_shape = self._get_shape_from_dict(self.y)

        if self.original_x_shape[0] != self.original_y_shape[0]:
            raise ValueError(f"X {self.original_x_shape} and Y {self.original_y_shape} must have "
                             f"the same number of samples (1st dimension)")

    def _get_shape_from_dict(self, d: Dict):
        """
        From dictionary of features get combined shape with common 1st dim and sub shapes as extra entries
        """

        shapes = tuple(
            v.shape[1:] if len(v.shape) > 1 else 1
            for v in d.values()
        )
        num_examples = list(d.values())[0].shape[0]
        return (num_examples,) + shapes

    def _initialized_input(self, value: Any):
        """Initialize an input based on the provided type"""
        if self.feed_as_dict:
            if isinstance(value, Dict):
                return self._initialize_from_dict(value)
            else:
                value = self._native_list_to_np_or_df(value)
                return self._dataframe_to_dict_of_series(value)
        else:
            return self._native_list_to_np_or_df(value)

    def _initialize_from_dict(self, value: Dict) -> Dict:
        """
        Initialize an input from a dictionary of feature columns
        """
        value = {
            feature: self._native_list_to_np_or_df(values)
            for feature, values in value.items()
        }
        return value

    def _dataframe_to_dict_of_series(self, ar: NpOrDataFrame) -> Dict[str, pd.Series]:
        if not isinstance(ar, pd.DataFrame):
            ar = pd.DataFrame(ar)

        return {
            col: ar[col]
            for col in ar.columns
        }

    def _native_list_to_np_or_df(self, ar: DataFrameCompat) -> NpOrDataFrame:
        """
        Convert a python native list to numpy for 1-D or pandas DataFrame for multi-dimensional or dict structure.

        :param ar: The array to convert
        :return: The converted array. If it is not a python native list it will return the original array.
        """
        if isinstance(ar, list):
            if isinstance(ar[0], (int, float)):
                return np.array(ar)
            else:
                return pd.DataFrame(ar)
        elif isinstance(ar, Dict):
            return pd.DataFrame(ar)
        else:
            return ar

    def get_input_fn(self, epochs: Optional[int] = None) -> InputFN:

        def _input_fn():
            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.x),
                tf.data.Dataset.from_tensor_slices(self.y)
            ))

            if self.shuffle:
                dataset = dataset.shuffle(self.original_x_shape[0])

            return dataset.repeat(count=epochs).batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        return _input_fn

    @abstractmethod
    def total_steps(self, epochs: int) -> int:
        if self.drop_remainder:
            return int(np.floor(epochs * self.original_x_shape[0] / self.batch_size))
        else:
            return int(np.ceil(epochs * self.original_x_shape[0] / self.batch_size))

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.original_x_shape}, " \
            f"y={self.original_y_shape}, batch_size={self.batch_size}, " \
            f"shuffle={self.shuffle}, as_dict={self.feed_as_dict})"

# class BatchGeneratorFeeder(FeederBase):
#     """
#     Feeder from pythonic batch generator
#     """
#
#     def __init__(self, generator_fn: Callable[[], Generator]):
#         self.generator_fn = generator_fn
#
#     def _inspect_generator_fn(self):
#         return_sample = next(self.generator_fn())
#         assert isinstance(return_sample, tuple)
#         assert len(return_sample) == 0
#
#         sample_x, sample_y = return_sample
#
#         print(return_sample)
#
#     def get_input_fn(self, epochs: Optional[int] = None) -> InputFN:
#         raise NotImplementedError()
#
#         def _input_fn():
#             return tf.data.Dataset.from_generator(self.generator_fn, )
#
#         return _input_fn
#
#     @abstractmethod
#     def total_steps(self, epochs: int) -> int:
#         raise NotImplementedError()
#
#
# class BatchSequenceFeeder(FeederBase):
#     """
#     Feeder for in memory batches
#     """
#
#     def __init__(self):
#         pass
#
#     def train_input(self):
#         for i in range(len(self)):
#             yield self[i]
#         raise StopIteration
#
#     @abstractmethod
#     def __getitem__(self, index):
#         raise NotImplementedError()
#
#     def __len__(self):
#         raise NotImplementedError()
