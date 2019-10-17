from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import json
from tqdm import tqdm


def _int64_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _string_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.encode('utf-8') for s in value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def file_generator(reader):
    """
    Convert a document into ids based on given vocabulary
    :param Iterator[str,int] reader: csv file reader iterator
    :rtype: Iterator[str,int]
    """
    for row_label, row_text in reader:
        yield row_label, row_text


def write_example_to_tfrecord(output_filename, row_generator, features_type='int_list'):
    """
    Reader for labeled text per row. The label is expected to be the first argument.
    :param Str output_filename: name of the output file
    :param Generator row_generator: Generator of lines to write to tfrecords
    :param Str features_type: type of features to be written in tfrecords
    """
    file_rows = 0
    with tf.python_io.TFRecordWriter(str(output_filename)+'.tfr') as tf_writer:
        for row_label, row_tokens in tqdm(row_generator):
            if features_type == 'string_list':
                features = _string_feature(row_tokens)
            else:
                features = _int64_feature(row_tokens)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'text': features,
                             'label': _int64_feature(row_label)}
                ))

            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            tf_writer.write(serialized)
            file_rows += 1
    return file_rows


def save_dict_to_json(d, json_path):
    """
    Reader for labeled text per row. The label is expected to be the first argument.
    :param Dict[str,int] d: dictionary to save
    :param Str json_path: path of the saved json
    """
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)
