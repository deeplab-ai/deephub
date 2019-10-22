import logging
from abc import abstractmethod
from typing import Union, Optional, Callable, List, Dict

import tensorflow as tf
import numpy as np
import pandas as pd

DataFrameCompat = Union[List[Dict], Dict[str, List], Dict[str, np.ndarray]]
NpOrDataFrame = Union[np.ndarray, pd.DataFrame]

logger = logging.getLogger(__name__)


class FeatureTypeBase:
    """
    Interface for higher-level features of tf.train.Example
    """

    @property
    @abstractmethod
    def raw_feature_type(self) -> tf.io.FixedLenFeature:
        """The native feature type that is used to store in tf_record"""
        raise NotImplementedError()

    @abstractmethod
    def post_parse_op(self, feature: tf.Tensor) -> tf.Tensor:
        """
        Transform raw feature to higher-level structure
        :param feature: The low level feature as it was parsed from tf_record
        :return: The higher level feature
        """
        raise NotImplementedError()


class ImageFeature(FeatureTypeBase):
    """
    Feature that can load encoded images from TF records
    """

    CHANNELS_ORIGINAL = 0
    CHANNELS_GRAYSCALE = 1
    CHANNELS_RGB = 3

    DECODE_FORMATS_TO_DECODER = {
        'jpeg': tf.image.decode_jpeg,
        'png': tf.image.decode_png,
        'gif': tf.image.decode_gif,
        'bmp': tf.image.decode_bmp,
        'auto': tf.image.decode_image
    }

    def __init__(self,
                 resize_height: Optional[int] = None,
                 resize_width: Optional[int] = None,
                 channels: int = CHANNELS_RGB,
                 decode_format: str = 'jpeg',
                 enforced_dtype: Optional[Union[str, tf.DType]] = tf.float32,
                 standard_scaling: bool = False,
                 post_op: Optional[Callable[[tf.Tensor], tf.Tensor]] = None):
        """
        Initialize image feature parser.

        :param resize_height: Must be set along with `resize_width` to resize image after loading.
        :param resize_width: Must be set along with `resize_height` to resize image after loading.
        :param channels: Use the constants CHANNELS_RGB, CHANNELS_GRAYSCALE,
            and CHANNELS_ORIGINAL to control the number of channels that the decoded image will have.
        :param decode_format: The format that image is encoded in protobuf. Supported formats are:
            'jpeg', 'png', 'gif', 'bmp' and 'auto. Notice that all but 'gif' return a 3D tensor while
            'gif' a 4D tensor. Also 'auto' returns unknown number of channels.
        :param enforced_dtype: If set it will enforce a new dtype of the output tensor.
        :param standard_scaling: If true it will perform a standard scaling and the output tensor will
            be in the range [-1, 1]
        :param
        :param post_op: Custom operation to perform at the final image tensor. This can be any function
            that receives a tf.Tensor and returns a transformed tf.Tensor.
        """

        if decode_format not in self.DECODE_FORMATS_TO_DECODER:
            raise ValueError(f"Unknown image decode format {decode_format}, "
                             f"currently supported are {list(self.DECODE_FORMATS_TO_DECODER)}")

        if (resize_width is None) ^ (resize_height is None):
            raise ValueError("You need to set both 'resize_width' and 'resize_height' "
                             "to perform resizing")
        if enforced_dtype is not None:
            enforced_dtype = tf.as_dtype(enforced_dtype)

        self.image_decoder = self.DECODE_FORMATS_TO_DECODER[decode_format]
        self.channels = channels
        self.enforced_dtype = enforced_dtype
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.standard_scaling = standard_scaling
        self.post_op = post_op

    @property
    def raw_feature_type(self):
        return tf.io.FixedLenFeature([], dtype=tf.string)

    def post_parse_op(self, feature: tf.Tensor):

        image = self.image_decoder(feature, channels=self.channels)

        if self.enforced_dtype is not None and self.enforced_dtype != image.dtype:
            image = tf.image.convert_image_dtype(image, dtype=self.enforced_dtype)

        if self.resize_height is not None and self.resize_width is not None:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [self.resize_height, self.resize_width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])

        if self.standard_scaling:
            image = (image - 0.5) * 2.0

        if self.post_op:
            image = self.post_op(image)

        return image
