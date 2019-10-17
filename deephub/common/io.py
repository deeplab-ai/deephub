from __future__ import absolute_import

import hashlib
from typing import List, Union, Optional
from pathlib import Path

import ujson as json

from tensorflow.python.platform import gfile

AnyPathType = Union[Path, str]


def append_to_txt_file(filename, content):
    """
    Append a string at the end of f
    :param Union[str, Path] filename: The path of the target file
    :param str content: The string to append to the text file
    """
    with open(str(filename), 'a') as f:
        f.write(content)


def append_to_json_file(filename, obj):
    """
    Append an object to json file as a new row
    :param  Union[str, Path] filename: The path of the target file
    :param Union[Dict, List] obj: The object to serialize in json format
    """
    json_str = json.dumps(obj)
    append_to_txt_file(filename, json_str + '\n')


def load_json_object(filename):
    """
    Load an object from a json file
    :param Union[str|Path] filename: The path of the source file
    :return: The deserialized object
    :rtype: Union[Dict, List]
    """
    with open(str(filename), 'r') as f:
        return json.load(f)


def file_md5(fname):
    """
    Calculate the md5 checksum of a file content
    :param Union[str, Path] fname: The filepath to calculate the md5 checksum
    :rtype: A string with hex encoded md5 sum
    """
    hash_md5 = hashlib.md5()
    with open(str(fname), "rb") as f:
        for chunk in iter(lambda: f.read(65535), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_filenames(glob_pattern):
    file_names = gfile.Glob(glob_pattern)
    if not file_names:
        raise ValueError('Found no files in --data_dir matching: {}'.format(glob_pattern))

    return file_names


def resolve_glob_pattern(pattern: str, starting_folder: Optional[AnyPathType] = None,
                         match_folders: bool = False) -> List[Path]:
    """
    Resolve a glob pattern and return a list of file paths
    :param pattern: The glob patter to resolve in file paths.
    :param starting_folder: The initial path to resolve glob pattern. If None then the current
    working directory will be used.
    :param match_folders: This flag controls if the return list will include folder paths.
        Setting this to to False it will return only matched file entries, but it will still traverse
        any folder that matches the pattern.
    :return: A list of resolved absolute file paths
    """

    pattern_path = Path(pattern)

    if pattern_path.is_absolute():
        if starting_folder:
            raise ValueError("You cannot define starting_folder if pattern is of absolute path.")
        starting_folder = Path(pattern_path.root)
        pattern = str(pattern_path.relative_to(starting_folder))
    else:
        if starting_folder:
            starting_folder = Path(starting_folder).resolve()
        else:
            starting_folder = Path('.').resolve()

    # Filter-out non directories
    matches = starting_folder.glob(pattern)
    if not match_folders:
        matches = filter(lambda p: not p.is_dir(), matches)

    return list(matches)
