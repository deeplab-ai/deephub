import logging

from pathlib import Path
from functools import partial
import yaml
import os

from deephub.common.logging import context_log
from deephub.common.gsutil import download_file, download_directory, recursive_size
from . import get_resources_writable_directory, get_resource_path

logger = logging.getLogger(__name__)
context_log = partial(context_log, logger=logger, logging_level=logging.INFO)


class RemoteResource(object):
    """
    Handler for managing remote resource stored in google storage
    """

    def __init__(self, name, dst_path, gs_url):
        """
        Initialize handler
        :param str name: The unique name of this resource
        :param str dst_path: The destination relative path in local repository
        :param str gs_url: The source url from google storage
        """
        self.name = name
        self.dst_path = dst_path
        self.gs_url = gs_url

    @property
    def is_directory(self):
        """
        Flag if this is resource is a directory of multiple files
        :rtype: bool
        """
        return self.dst_path.endswith('/')

    def is_file(self):
        """
        Flag if this resource is a single file
        :rtype: bool
        """
        return not self.is_directory

    def local_size(self):
        """
        Calculate the storage size in local repository
        :rtype: int
        """
        local_path = get_resource_path(self.dst_path)
        if not local_path.exists():
            return 0  # This resource is not downloaded
        raise NotImplementedError()

    def remote_size(self):
        """
        Calculate the storage in remote repository
        :rtype: int
        """
        return recursive_size(self.gs_url)

    def download(self):
        """
        Download this resource in local file storage
        """
        with context_log(f"Downloading remote resource '{self.name}'"):
            if self.is_directory:
                dst_directory = get_resources_writable_directory(self.dst_path)
                download_directory(self.gs_url, dst_directory)
            else:
                # Download single file
                dst_directory = get_resources_writable_directory(Path(self.dst_path).parent)
                dst_fname = Path(self.dst_path).name

                download_file(self.gs_url, dst_directory / dst_fname)


# Registry of remote resources that are downloadable from external storages.
_remote_resources_registry = {}


def register_remote_resource(*args, **kwargs):
    """
    Register a new resource in registry.

    The handler will be created from given arguments. See RemoteResource for more details.
    :return: The constructed remote resource object
    :rtype: RemoteResource
    """
    global _remote_resources_registry
    resource = RemoteResource(*args, **kwargs)

    if resource.name in _remote_resources_registry:
        raise KeyError("There is already a resource registered with that name")

    _remote_resources_registry[resource.name] = resource
    return resource


def get_resource(resource_id):
    """
    Get handler to a registered resource
    :param str resource_id: The unique resource name
    :rtype: RemoteResource
    """
    return _remote_resources_registry[resource_id]


def get_all_resources():
    """
    Get handlers of all registered remote resources
    :rtype: typing.List[RemoteResource]
    """
    return _remote_resources_registry.values()


def load_resources_from_yaml(filepath=None):
    """
    Load resources in registry from external yaml file
    :param Union[str, Path] filepath: The path of YAML
    :rtype: None
    """
    if filepath is None:
        filepath = Path(__file__).parent / 'remote.yaml'

    if os.path.exists(filepath):
        logger.debug(f"Loading remote resources configuration from '{filepath}'")
        with open(str(filepath), 'rt') as f:
            remote_resources = yaml.load(f)

        for resource in remote_resources['resources']:
            register_remote_resource(**resource)
    else:
        logger.warning("The resources config file ({}) does not exists.".format(filepath))


# Load all resources from external file
load_resources_from_yaml()
