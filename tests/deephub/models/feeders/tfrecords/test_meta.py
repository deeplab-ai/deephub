import pytest
import shutil
import io
import os
from pathlib import Path

import tensorflow as tf

from deephub.models.feeders.tfrecords.meta import (TFRecordMetadata,
                                                         get_fileinfo, generate_fileinfo, TFRecordInfoMissingError,
                                                         TFRecordValidationError)


class TestPublicApiMeta:
    def test_get_info_non_existing_file(self, tmpdir):
        with pytest.raises(FileNotFoundError):
            get_fileinfo(Path(tmpdir / 'nonexisting.tfrecords'))

        with pytest.raises(FileNotFoundError):
            generate_fileinfo(Path(tmpdir / 'nonexisting.tfrecords'))

    def test_non_existing_meta(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        with pytest.raises(TFRecordInfoMissingError):
            get_fileinfo(Path(tmpdir / 'train-00000-of-00001'))

    def test_non_existing_meta_file_generate(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)

        fpath = Path(tmpdir / 'train-00000-of-00001')

        # Try to get on folder without metadata
        with pytest.raises(TFRecordInfoMissingError):
            get_fileinfo(fpath)

        # Try to generate
        info = generate_fileinfo(fpath)
        assert info.md5_hash == '3c8c216b7293fdef623b04e01bb5878a'
        assert info.file_size == 350
        assert info.name == 'train-00000-of-00001'
        assert info.full_path == Path(fpath)

        # Try again to fetch from generated metadata
        info2 = get_fileinfo(fpath)

        assert info is not info2
        assert info == info2

    def test_invalid_size(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        fpath = Path(tmpdir / 'train-00000-of-00001')

        # Generate info
        original_info = generate_fileinfo(fpath)

        # Change a bit the size
        with open(fpath, 'ab') as f:
            f.write('junk'.encode('utf-8'))

        with pytest.raises(TFRecordValidationError):
            get_fileinfo(fpath, shallow_check=False)

        # Try to regenerate, will not fail because tf can handle trailing rubbish
        info2 = generate_fileinfo(fpath)
        assert original_info is not info2
        assert original_info != info2
        assert info2.file_size == 354
        assert info2.md5_hash == '76a086a01e382560309ccfc232711dec'

    def test_invalid_hash(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        fpath = Path(tmpdir / 'train-00000-of-00001')

        # Generate info
        original_info = generate_fileinfo(fpath)

        # Change a bit the content by writing random data at the beginning
        with open(fpath, 'r+b') as f:
            f.seek(0, io.SEEK_SET)
            print(f.tell())
            f.write('junk'.encode('utf-8'))

        with pytest.raises(TFRecordValidationError):
            get_fileinfo(fpath, shallow_check=False)

        # Try to regenerate, should fail because not valid tf records
        with pytest.raises(tf.errors.DataLossError):
            generate_fileinfo(fpath)

    def test_deleted(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        fpath = Path(tmpdir / 'train-00000-of-00001')

        # Generate info
        original_info = generate_fileinfo(fpath)

        # Remove the file
        os.unlink(fpath)

        with pytest.raises(FileNotFoundError):
            get_fileinfo(fpath, shallow_check=False)

    def test_generation_multiple_file_info(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'validation-00000-of-00001', tmpdir)

        fpath1 = Path(tmpdir / 'train-00000-of-00001')
        fpath2 = Path(tmpdir / 'validation-00000-of-00001')

        # Generate for both files
        original_info1 = generate_fileinfo(fpath1)
        original_info2 = generate_fileinfo(fpath2)

        assert original_info1.md5_hash == '3c8c216b7293fdef623b04e01bb5878a'
        assert original_info2.md5_hash == 'fdebe01f545d90f127a15ea2f28d3d1d'

        # Get from stored metadata
        info1 = get_fileinfo(fpath1)
        info2 = get_fileinfo(fpath2)

        assert info1 is not original_info1
        assert info1 == original_info1

        assert info2 is not original_info2
        assert info2 == original_info2


class TestTFRecordMetaData:
    def test_contains_iter_get(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'validation-00000-of-00001', tmpdir)

        fpath1 = Path(tmpdir / 'train-00000-of-00001')
        fpath2 = Path(tmpdir / 'validation-00000-of-00001')

        # Generate for both files
        original_info1 = generate_fileinfo(fpath1)
        original_info2 = generate_fileinfo(fpath2)

        meta = TFRecordMetadata.from_folder(fpath1.parent)

        assert 'train-00000-of-00001' in meta
        assert 'validation-00000-of-00001' in meta
        assert 'unknown' not in meta

        info1 = meta['train-00000-of-00001']
        assert info1.name == 'train-00000-of-00001'

        info2 = meta['validation-00000-of-00001']
        assert info2.name == 'validation-00000-of-00001'
