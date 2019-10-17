from pathlib import Path
from typing import Dict, Any
import json
import logging

import attr
import tensorflow as tf

from deephub.common.io import file_md5

METADATA_FILENAME = '_tfrecord_metadata.json'
CURRENT_VERSION = 1

logger = logging.getLogger(__name__)


class TFRecordMetadataError(RuntimeError):
    """Generic error with tf record metadata """


class TFRecordValidationError(TFRecordMetadataError):
    """Exception raised if the tfrecord file has different checksum"""


# class TFRecordMetadataFileMissingError(TFRecordMetadataError):
#     """Exception raised when no metadata file was found in folder."""


class TFRecordMetadataFileFormatError(TFRecordMetadataError):
    """Exception raised when metadata file is wrongly formatted."""


class TFRecordInfoMissingError(TFRecordMetadataError):
    """Exception raised when file was not found in metadata file."""


@attr.s()
class TFRecordFileInfo:
    """
    Wrapper class for single tfrecords file metadata
    """

    full_path: Path = attr.ib()
    md5_hash: str = attr.ib()
    total_records: int = attr.ib()
    file_size: int = attr.ib()
    meta: int = attr.ib(factory=dict)

    @classmethod
    def generate_for(cls, fpath: Path) -> 'TFRecordFileInfo':
        """
        Generate initial metadata from an existing tf record file
        :param fpath: The path to the file

        """
        file_size = fpath.stat().st_size
        meta_info = cls(full_path=fpath.resolve(),
                        md5_hash=file_md5(fpath),
                        total_records=sum(1 for _ in tf.python_io.tf_record_iterator(path=str(fpath))),
                        file_size=file_size
                        )

        logger.debug(f"Generated meta info {meta_info} for file fpath")
        return meta_info

    @property
    def name(self) -> str:
        """Get the name of the file that this metadata corresponds to"""
        return self.full_path.name

    def validate(self, shallow_check: bool):
        """
        Check that this file is unchanged based on the checksum stored in metadata
        :param shallow_check: Determines if the md5 hash will be checked.
        :raises TFRecordWrongHashError
        """
        if not self.full_path.exists() or not self.full_path.is_file():
            raise TFRecordValidationError(f"File {self.full_path} does not exist any more.")

        current_size = self.full_path.stat().st_size
        if current_size != self.file_size:
            raise TFRecordValidationError(f"File {self.full_path} has changed size from "
                                          f"{self.file_size} to {current_size}")

        if shallow_check is False and file_md5(self.full_path) != self.md5_hash:
            raise TFRecordValidationError(f"File {self.full_path} does not have same hash in metadata")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_records': self.total_records,
            'file_size': self.file_size,
            'md5_hash': self.md5_hash
        }


@attr.s()
class TFRecordMetadata:
    """
    Wrapper for container of tfrecords metadata
    """

    folder: Path = attr.ib()
    files: Dict[str, TFRecordFileInfo] = attr.ib(factory=dict)
    version: int = attr.ib(default=CURRENT_VERSION)

    @classmethod
    def _load_from_file(cls, fpath: Path):
        """Load metadata from file"""

        with open(fpath, 'rt') as f:
            try:
                document = json.load(f)
            except Exception as e:
                raise TFRecordMetadataFileFormatError(f"Error while parsing json from metadata file {e!s}")

        if document.get('version') != 1:
            raise TFRecordMetadataFileFormatError(
                f"Do not know how to handle metadata of version {document.get('version')}")

        file_infos = {
            fname: TFRecordFileInfo(
                full_path=fpath.parent / fname,
                md5_hash=file_params['md5_hash'],
                total_records=file_params['total_records'],
                file_size=file_params['file_size']
            )
            for fname, file_params in document.get('files', {}).items()
        }
        return cls(
            folder=fpath.parent,
            files=file_infos,
            version=document.get('version')
        )

    @classmethod
    def from_folder(cls, path: Path) -> 'TFRecordMetadata':
        """
        Open metadata of a folder
        :param path: The path to the folder
        :return: The loaded wrapper of metadata information
        """

        metadata_fname = path / METADATA_FILENAME
        if not metadata_fname.is_file():
            raise TFRecordInfoMissingError(f"Cannot find metadata file for folder: {path}")

        try:
            return cls._load_from_file(metadata_fname)
        except Exception as e:
            logger.error(f"Error while loading metadata file form folder {path}: {e!s}")
            raise

    def __contains__(self, filename: str) -> bool:
        """Check that a filename is contained in metadatas file"""
        return filename in self.files

    def __getitem__(self, filename: str):
        """
        Get info for a specific filename
        :raises TFRecordInfoMissingError if the file is not found in metadata
         """
        if filename not in self.files:
            raise TFRecordInfoMissingError(f"There are no metainfo for file {filename},"
                                           f"you need to manually generate them.")
        return self.files[filename]

    def __setitem__(self, filename: str, info: TFRecordFileInfo) -> TFRecordFileInfo:
        """
        Set info for a specific filename
        :param filename: The filename to set info for.
        :param info: The info to write in metadata file
        """
        if info.name != filename:
            raise ValueError(f"{info} is not for file {filename}")
        self.files[filename] = info
        return info

    def flush(self):
        """Flush info on disk file"""

        document = {
            'version': self.version,
            'files': {
                fname: info.to_dict()
                for fname, info in self.files.items()
            }
        }
        with open(self.folder / METADATA_FILENAME, 'wt') as f:
            json.dump(document, f, indent=4)


def get_fileinfo(fpath: Path, shallow_check: bool = True) -> TFRecordFileInfo:
    """
    Request info for a specific filename
    :param fpath: The filename path to load info for.
    :param shallow_check: True/False validation. With shallow_check=True only the file size will be validated
                          while with shallow_check=False both size and md5 hash will be checked for the tf_record file.
    :return The record file information
    """
    if not fpath.is_file():
        raise FileNotFoundError(f"Cannot access file {fpath}")
    metadata = TFRecordMetadata.from_folder(fpath.parent)
    info = metadata.files[fpath.name]
    info.validate(shallow_check)
    return info


def generate_fileinfo(fpath: Path, always_regenerate: bool = False) -> TFRecordFileInfo:
    """
    Generate file info for a file if it does not exist.
    :param fpath: The path of the filename to generate info
    :param always_regenerate: If true it will regenerate info even if they file seems unchanged.
    :return The new info of the file
    """

    # Get folder container
    try:
        metadata = TFRecordMetadata.from_folder(fpath.parent)
    except Exception:
        # Generate a new file
        metadata = TFRecordMetadata(folder=fpath.parent)

    # Get file info or generate
    try:
        info = metadata[fpath.name]
        if always_regenerate:
            raise TFRecordValidationError()
        info.validate(shallow_check=False)
    except (TFRecordInfoMissingError, TFRecordValidationError):
        info = TFRecordFileInfo.generate_for(fpath)

    # Update metadata
    metadata[fpath.name] = info

    # Write metadata to storage
    metadata.flush()

    return info
