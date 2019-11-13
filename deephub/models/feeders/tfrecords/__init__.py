import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, NamedTuple, Tuple
from functools import lru_cache
import multiprocessing

import tensorflow as tf
import numpy as np

from deephub.common.modules import instantiate_from_dict
from deephub.common.io import AnyPathType
from deephub.common.utils import merge_dictionaries
from deephub.common.io import resolve_glob_pattern
from .features import FeatureTypeBase
from ..base import FeederBase, InputFN
from .meta import get_fileinfo, generate_fileinfo


logger = logging.getLogger(__name__)


class TFRecordExamplesFeeder(FeederBase):
    """
    Feeder for tf.train.Examples stored in TFRecord files.

    The TFRecordExamplesFeeder is capable to read streams
    of tf_records, parse them using the tf.train.Example proto
    and convert them to feature-label dataset. The user has
    to provide the features that are stored inside the examples
    mapped by a unique key and the keys that will be outputed
    as training features and labels.

    """

    def __init__(self,
                 file_patterns: Union[AnyPathType, List[AnyPathType]],
                 features_map: Dict[str, Union[NamedTuple, FeatureTypeBase]],
                 max_examples: int = -1,
                 labels_map: Optional[Dict[str, Union[NamedTuple, FeatureTypeBase]]] = None,
                 batch_size: int = 128,
                 drop_remainder: bool = False,
                 shuffle: bool = True,
                 shuffle_buffer_size: Optional[int] = 10000,
                 num_parallel_maps: Optional[int] = None,
                 num_parallel_reads: Optional[int] = 32,
                 prefetch: Optional[int] = None,
                 auto_generate_meta: bool = False):
        """
        Initialize the feeder

        :param file_patterns: A glob file pattern to local file system. It also support a list of patterns
            and will read the union of all matched files.
        :param features_map: A dictionary of example features that will be used as input features in the model. The
            dictionary consists of standard feature descriptors such as tf.io.FixedLenFeature or by custom
            FeatureTypeBase objects, that are mapped by the name that was used to serialize them inside examples.
        :param max_examples: Number of examples to load from the dataset. Default value -1 will load the
            whole dataset with total_examples rows. If shuffling is enabled, it will happen on the truncated dataset.
        :param labels_map: A dictionary of example features that will be used as input labels in the model. The
            dictionary follows the same format as in `features_map` argument.
        :param batch_size: The size of the mini-batch that will produce this feeder
        :param drop_remainder: Determines if you drop or not the remainder of data rows (total examples%batch_size)
        :param shuffle: A flag to control if input should be shuffled
        :param shuffle_buffer_size: Buffer size for the shuffle data operation
        :param num_parallel_maps: The number of parallel workers to perform the mapping/parsing on examples.
            Setting this to None, will try to detect a good value for optimal efficiency
        :param num_parallel_reads: The number of files to read in parallel.
        :param prefetch: The number of batches that will be buffered when prefetching.
            Setting this to None, will try to detect a good value for optimal efficiency
        :param auto_generate_meta: If true the feeder will generate metadata for each tfrecord that does not
            have. WARNING: This process may take a long time depending the size of the file and you need to make
            sure that you have write access to the same folder as the tfrecord files.
        """
        if not isinstance(file_patterns, (list, tuple)):
            file_patterns = (file_patterns,)
        self.file_patterns = file_patterns
        self.auto_generate_meta = auto_generate_meta

        if not self.file_paths:
            raise FileNotFoundError(f"No file matched with provided file patterns {self.file_patterns}. ")

        self.max_examples = max_examples

        if batch_size <= 0:
            raise ValueError('Batch size must be a positive number')

        if batch_size > self.total_examples:
            raise ValueError(f"The batch size({batch_size}) is greater than the max_examples({self.total_examples}) "
                             f"loaded from the dataset. Please set this value in range between "
                             f"[1, {self.total_examples}].")

        self.batch_size = batch_size

        if num_parallel_maps is None:
            num_parallel_maps = multiprocessing.cpu_count()
            logger.info(f"TFRecordExamplesFeeder choosing '{num_parallel_maps}' parallel calls for map() "
                        f"based on discovered CPUs")

        logger.info(f"Feeding {self.total_examples} total examples.")

        self.shuffle_buffer_size = min(shuffle_buffer_size, self.total_examples)

        self.features_map = self._construct_features_map(features_map=features_map)
        if labels_map is not None:
            self.labels_map = self._construct_features_map(features_map=labels_map)
        else:
            self.labels_map = None

        self.shuffle = shuffle
        self.num_parallel_reads = num_parallel_reads
        self.num_parallel_maps = num_parallel_maps
        self.prefetch = prefetch
        self.drop_remainder = drop_remainder

    @property
    @lru_cache()
    def file_paths(self) -> List[Path]:
        """The list of files that matched the given patterns"""

        fpaths = []
        for pattern in self.file_patterns:
            fpaths.extend(resolve_glob_pattern(pattern, match_folders=False))

        return fpaths

    @property
    @lru_cache()
    def total_examples(self) -> Optional[int]:
        """The total number of examples that was found in input files"""
        if self.auto_generate_meta:
            info_f = generate_fileinfo
        else:
            info_f = get_fileinfo

        total_examples = sum(
            info_f(fpath).total_records
            for fpath in self.file_paths
            )

        if self.max_examples > total_examples:
            raise ValueError("Feeder input max_examples({}) is greater than the "
                             "total({}) number of samples into the dataset. Your "
                             "calculations about steps per epoch and other metrics may be"
                             "incorrect, so please change the max_examples to a "
                             "correct value.".format(self.max_examples, total_examples))

        if self.max_examples == -1:
            logger.warning("Input max_examples equal to -1, means that the whole dataset with size ({}) will be loaded"
                           .format(total_examples))
        else:
            total_examples = self.max_examples

        return total_examples

    @classmethod
    def _construct_features_map(
            cls,
            features_map: Dict[str, Union[Dict, FeatureTypeBase]]) -> Dict[str, FeatureTypeBase]:
        """
        Create an example feature based on a dictionary that describes the type and parameters.

        The reconstruction process is similar to reconstruction followed by variant definition for
        Model and Feeders

        :param features_map: Example features or dictionary with their definitions mapped in unique key.
        :return: The example feature objects mapped in their key, preserving the same order as in `features_map`
        """

        features_map_objects = {}

        if 'module_path' not in features_map:
            module_path = 'tensorflow'
        else:
            module_path = features_map['module_path']
            del (features_map['module_path'])

        for fname, fmap in features_map.items():

            if not isinstance(fmap, dict):
                features_map_objects[fname] = fmap
                continue

            features_map_objects[fname] = instantiate_from_dict(
                fmap,
                search_modules=[module_path])

        result = features_map_objects
        return result

    def _parse_example(self, raw_record: tf.Tensor) -> Union[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
                                                             Dict[str, tf.Tensor]]:
        """
        Parse a single example and return a tuple with features and labels mapped by their key name

        :param raw_record: The raw tf_record as extracted from the stream
        """
        if self.labels_map is not None:
            example_features = merge_dictionaries(self.features_map, self.labels_map)
        else:
            example_features = self.features_map

        # Override custom-features with raw feature type
        custom_features = {
            name: feature
            for name, feature in example_features.items()
            if hasattr(feature, 'raw_feature_type')
        }
        for name in custom_features.keys():
            example_features[name] = custom_features[name].raw_feature_type

        # Native example parsing
        example = tf.parse_single_example(serialized=raw_record, features=example_features)

        # Post-process custom feature types
        for name, custom_feature in custom_features.items():
            example[name] = custom_feature.post_parse_op(example[name])

        if self.labels_map is not None:
            return (
                {key: example[key] for key in self.features_map.keys()},
                {key: example[key] for key in self.labels_map.keys()}
            )
        else:
            return {key: example[key] for key in self.features_map.keys()}

    def get_input_fn(self, epochs: Optional[int] = None) -> InputFN:

        def _input_fn():
            return self.get_tf_dataset(epochs=epochs)

        return _input_fn

    def get_tf_dataset(self, epochs: Optional[int] = None) -> tf.data.Dataset:
        if self.max_examples == -1:
            raw_records = tf.data.TFRecordDataset(list(map(str, self.file_paths)),
                                                  num_parallel_reads=self.num_parallel_reads)
        else:
            raw_records = tf.data.TFRecordDataset(list(map(str, self.file_paths)),
                                                  num_parallel_reads=self.num_parallel_reads). \
                take(count=self.total_examples)
        examples = raw_records.map(self._parse_example, num_parallel_calls=self.num_parallel_maps)
        if self.shuffle:
            examples = examples.shuffle(self.shuffle_buffer_size, reshuffle_each_iteration=True).repeat(count=epochs)
        else:
            examples = examples.repeat(count=epochs)
        batches = examples.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)
        # buffer_size=None is autotuning the appropriate prefetching buffer size
        return batches.prefetch(buffer_size=self.prefetch)

    def total_steps(self, epochs: int) -> Optional[int]:
        if self.total_examples is None:
            return None
        else:
            if self.drop_remainder:
                return int(np.floor((self.total_examples / self.batch_size) * epochs))
            else:
                return int(np.ceil((self.total_examples / self.batch_size) * epochs))
