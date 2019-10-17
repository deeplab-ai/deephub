import pytest
import tensorflow as tf

from deephub.models.feeders.tfrecords.features import ImageFeature


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
