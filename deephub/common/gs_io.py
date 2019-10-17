from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

from urllib.parse import urlsplit
from pathlib import Path
from google.cloud import storage as gstorage

logger = logging.getLogger(__name__)


def create_storage_client():
    """
    Create a new google storage client
    :rtype: gstorage.Client
    """
    storage_client = gstorage.Client()
    return storage_client


def parse_gs_url(gs_url):
    """
    Parse Google Storage url in primitive components
    :param str gs_url: Google storage url such as gs:/bucket/path/...
    :return The bucket and the path of the referenced object
    :rtype: Tuple[str, str]
    """
    parsed = urlsplit(gs_url)
    if parsed.scheme.lower() != 'gs':
        raise ValueError("This is not a google storage uri: '{}'".format(gs_url))
    bucket, path = parsed.netloc, parsed.path.lstrip('/')
    return bucket, path


def download_file(gs_url, dst_fpath, client=None):
    """
    Download a file from GS storage and save it in local path
    :param str gs_url: The source url in GS storage to download file from
    :param Union[str, Path] dst_fpath: The target file path to store the downloaded file
    :param google.cloud.storage.Client client: The GS Client to use for downloading file. If None a new client will
    be constructed.
    """

    if client is None:
        client = create_storage_client()
    bucket_name, path = parse_gs_url(gs_url)

    bucket = client.get_bucket(bucket_name)
    bucket.blob(path).download_to_filename(str(dst_fpath))


def download_directory(gs_url, dst_path, client=None):
    """
    Download a directory from GS storage and save it in local path
    :p
    :param str gs_url: The source url in GS storage to download file from
    :param Union[str, Path] dst_fpath: The target file path to store the downloaded file
    :param google.cloud.storage.Client client: The GS Client to use for downloading file. If None a new client will
    be constructed.
    """

    if client is None:
        client = create_storage_client()

    bucket_name, root_path = parse_gs_url(gs_url)

    bucket = client.get_bucket(bucket_name)

    # List blobs and download
    for blob in bucket.list_blobs(prefix=root_path):
        common_prefix = os.path.commonprefix([root_path, blob.name])
        if common_prefix != root_path:
            raise RuntimeError("There was an error downloading blob: '{}'".format(blob.name))

        relative_path = blob.name[len(common_prefix):].lstrip('/')

        dst_fpath = Path(dst_path) / relative_path
        blob.download_to_filename(str(dst_fpath))
