import re
from contextlib import contextmanager
import subprocess
from logging import getLogger

logger = getLogger(__name__)


class GSUtilMissingError(OSError):
    """
    Error occurred when gsutil is missing from operating system
    """
    pass


class GSUtilGeneralError(OSError):
    """
    Exception raised when gsutil has not been initialized with proper user
    """
    pass


@contextmanager
def inform_for_missing_gsutil():
    try:
        yield
    except OSError:
        raise GSUtilMissingError(
            "This operation depends on gsutil CLI tool.\n"
            "Please install the google cloud SDK (https://cloud.google.com/sdk/) and follow "
            "the instructions to login with your account before proceeding.")
    except subprocess.CalledProcessError:
        raise GSUtilGeneralError(
            "There was an error while gsutil tried to contact google server.\n"
            "Please check that you have been properly authenticated or that you don't have connection issues")


def download_file(gs_url, dst_fpath):
    """
    Download a file from google storage to local filesystem using the CLI tool `gsutil`
    :param str gs_url: The source path in google storage
    :param Union[str, Path] dst_fpath: The destination path in local filesystem
    """
    logger.debug(f"Downloading remote file '{gs_url}' at '{dst_fpath}'")
    with inform_for_missing_gsutil():
        subprocess.check_call([
            'gsutil',
            'cp',
            gs_url,
            str(dst_fpath)
        ],
            shell=False)


def download_directory(gs_url, dst_path):
    """
    Download recursively a directory from google storage to local filesystem using the CLI tool `gsutil`
    :param str gs_url: The source path in google storage
    :param Union[str, Path] dst_path: The destination directory in local filesystem
    """
    logger.debug(f"Downloading remote directory '{gs_url}' at '{dst_path}'")
    with inform_for_missing_gsutil():
        subprocess.check_call([
            'gsutil',
            '-m',
            'rsync',
            '-r',
            '-c',
            str(gs_url),
            str(dst_path)
        ],
            shell=False)


def recursive_size(gs_url):
    """
    Calculate the size of all object in the specified bucket with the same prefix.

    This command is similar to POSIX `du` command
    :param str gs_url: The bucket an path to search for object
    :rtype: int
    """
    logger.debug(f"Requesting storage size of remote resource '{gs_url}'")
    with inform_for_missing_gsutil():
        output = subprocess.check_output([
            'gsutil',
            'du',
            '-s',
            gs_url
        ],
            shell=False)

    matches = re.match(r'^(?P<size>\d+)\s+.+', output.decode("utf-8"))

    return int(matches.groupdict()['size'])
