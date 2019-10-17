from __future__ import absolute_import

import logging
import os
import re
from typing import Dict

import pandas as pd

NAMES_TO_CONVERT_TO_STRING = ['item_id']
SOME_CHAR = '#'
TOKENIZER = re.compile(r'\w+|[^\w\s]', re.UNICODE)

logger = logging.getLogger(__name__)


def _create_special_marks_sequences_regex(max_seq_length=4):
    """
    Create a regex object which can detect sequences of '.'/'!'/'?'.
    It's assumed the regex will be fed with the result of the tokenization process followed by a space,
    i.e. it can detect sequences of the form '. . . ! ? ! '.
    :param max_seq_length: the regex match result will contain two groups: the first group will match the
           first max_seq_length chars of a sequences (in the previous example, given max_seq_length == 4: '. . . ! '),
           and the second group will match the rest of the chars.
           Why 4 by default? the longest legitimate sequence I can think of is '?...'
    """
    special_mark = r'(?:[.!?] )'
    return re.compile('(' + special_mark + '{1,' + str(max_seq_length) + '})(' + special_mark + '*)', re.UNICODE)


SPECIAL_MARKS_SEQUENCES = _create_special_marks_sequences_regex()


def tokenize_title(title):
    if title is None or pd.isnull(title) or title.strip() == "":
        return []
    # split to tokens:
    tokenized = TOKENIZER.findall(title.lower())
    # truncate any series of ./?/!
    # this is needed since longer sequences might be rare, and thus will be filtered out
    tokenized = ' '.join(tokenized) + ' '
    tokenized = re.sub(SPECIAL_MARKS_SEQUENCES, lambda match: match.groups()[0].replace(' ', '') + ' ',
                       tokenized).strip()
    return tokenized.split(' ')


def string_to_bool(value):
    value = str(value).lower().strip()
    if value == 'true':
        return True
    if value == 'false':
        return False
    else:
        err_msg = '{} is not a valid boolean'.format(value)
        raise ValueError(err_msg)


def get_config_from_env_vars(suffix):
    env_configs = dict()

    env_vars = os.environ.copy()
    for config, value in env_vars.items():
        config = config.lower().strip()
        if config.startswith(suffix + '_'):
            config = config[3:]
            value = value.strip()
            if value == 'None':
                value = None
            env_configs[config] = value

    if not env_configs:
        logger.info('no configurations found in env variables')
    else:
        logger.info('configurations found in env variables: %s', env_configs)

    return env_configs


def size_formatted(bytes_size, suffix='B'):
    """
    Get a human readable representation of storage size

    Note: Taken from
        https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    :param int bytes_size: The size of object in bytes
    :rtype: str
    """
    if bytes_size < 1024:
        return "%3dB" % (bytes_size)
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(bytes_size) < 1024.0:
            return "%3.1f%s%s" % (bytes_size, unit, suffix)
        bytes_size /= 1024.0
    return "%.1f%s%s" % (bytes_size, 'Yi', suffix)


def merge_dictionaries(first, *others):
    """
    Merge multiple dictionaries in one.

    When multiple dictionaries have the same key, then the value of the latest (right most) will be selected.
    :param Dict first: The first dictionary
    :param List[Dict] others: The rest of dictionaries to be merged in one.
    :return: One new dictionary with the union of all keys
    :rtype: Dict
    """
    out = first.copy()

    for other in others:
        out.update(other)
    return out


def deep_update_dict(first: Dict, other: Dict) -> Dict:
    """
    Perform in place deep update of the first dictionary with the values of second
    :param first: The dictionary to be updated
    :param other: The dictionary to read the updated values
    :return: A reference to first dictionary
    """

    for k, v in other.items():

        if isinstance(first.get(k), dict) and isinstance(v, dict):
            # Both entries are dict type and we need to perform deep_update
            deep_update_dict(first[k], v)
            continue

        first[k] = v

    return first
