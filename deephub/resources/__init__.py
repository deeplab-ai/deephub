from __future__ import unicode_literals
from __future__ import absolute_import

import os

from pathlib import Path


class ResourceNotFound(OSError):
    pass


DEFAULT_USER_RESOURCES_DIRECTORY = 'runtime_resources'

_default_package_dir = Path(__file__).resolve().parent / 'blobs'
_user_resources_dir = Path(os.getcwd()).resolve() / DEFAULT_USER_RESOURCES_DIRECTORY


def set_user_resources_directory(new_path):
    """
    Change the path where to read for user specific resources
    :param Union[str, Path] new_path: The new path to search for user resources
    """
    global _user_resources_dir
    _user_resources_dir = Path(new_path)


def get_resource_path(*subpaths):
    """
    Get the absolute path of a resource file. Resources are first searched in the local working directory
    and then on the application package.

    :param str subpaths: The relative path inside the resources folder
    :rtype: Path
    """
    subpaths = [
        path.strip('/')
        for path in subpaths
    ]

    relative_path = os.path.join(*subpaths)

    if (_user_resources_dir / relative_path).exists():
        return _user_resources_dir / relative_path
    elif (_default_package_dir / relative_path).exists():
        return _default_package_dir / relative_path
    else:
        raise ResourceNotFound("Cannot find resource \"{}\"".format(relative_path))


def get_resources_writable_directory(*subpaths):
    """
    Get the absolute path to a writable directory in runtime path. Any intermediate directory will be first created.

    :param str|Path subpaths:
    :return: The absolute path in local storage
    :rtype: Path
    """

    subpaths = [
        str(path).strip('/')
        for path in subpaths
    ]

    relative_path = os.path.join(*subpaths)

    full_path = _user_resources_dir / relative_path

    # Assure that the directory exists
    if not os.path.exists(str(full_path)):
        os.makedirs(str(full_path))

    return full_path
