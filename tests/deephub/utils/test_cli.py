import os
from pathlib import Path
import shutil
import time
import click.testing

from deephub.models.feeders.tfrecords.meta import generate_fileinfo, get_fileinfo, TFRecordValidationError, \
    TFRecordInfoMissingError
from deephub.utils.__main__ import cli


class TestUtilsCLI:

    def _invoke_cli(self, *args):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args=args)

    def test_generate(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'validation-00000-of-00001', tmpdir)

        result = self._invoke_cli('generate-metadata', str(tmpdir / '*'))
        assert result.exit_code == 0
        assert "2 files matched" in result.output
        assert "Finished" in result.output

        info = get_fileinfo(Path(tmpdir / 'train-00000-of-00001'))
        assert info.file_size == 350
        assert info.total_records == 10
        assert info.md5_hash == '3c8c216b7293fdef623b04e01bb5878a'

        info = get_fileinfo(Path(tmpdir / 'validation-00000-of-00001'))
        assert info.file_size == 350
        assert info.total_records == 10
        assert info.md5_hash == 'fdebe01f545d90f127a15ea2f28d3d1d'

    def test_total_examples(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'validation-00000-of-00001', tmpdir)

        fpath = Path(tmpdir / 'train-00000-of-00001')
        generate_fileinfo(fpath)

        result = self._invoke_cli('total-examples', str(tmpdir / 'train-*'))
        assert 'Total number of examples: 10' in result.output

        fpath = Path(tmpdir / 'validation-00000-of-00001')
        generate_fileinfo(fpath)

        result = self._invoke_cli('total-examples', str(tmpdir / '*'))
        assert 'Total number of examples: 20' in result.output

    def test_shallow_deep_validate(self, tmpdir, datadir):
        for _ in range(10 ** 4):
            with open(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', 'rb') as fr:
                with open(tmpdir / 'train-00000-of-00001', 'ab') as fw:
                    fw.write(bytes(fr.read()))

        result = self._invoke_cli('generate-metadata', str(tmpdir / '*'))
        assert result.exit_code == 0

        start_time = time.time()
        result = self._invoke_cli('validate', str(tmpdir / '*'), '--shallow-check')
        assert result.exit_code == 0
        end_time = time.time()
        time1 = end_time - start_time

        start_time = time.time()
        result = self._invoke_cli('validate', str(tmpdir / '*'), '--deep-check')
        assert result.exit_code == 0
        end_time = time.time()
        time2 = end_time - start_time

        assert time1 < time2, f'i: {i}'

    def test_validate(self, tmpdir, datadir):
        shutil.copy(datadir / 'tfrecords' / 'simple' / 'train-00000-of-00001', tmpdir)
        result = self._invoke_cli('generate-metadata', str(tmpdir / '*'))
        assert result.exit_code == 0

        with open(tmpdir / 'train-00000-of-00001', 'rb') as fr:
            with open(tmpdir / 'train-00000-of-00001', 'ab') as fw:
                fw.write(bytes(fr.read()))

        result = self._invoke_cli('validate', str(tmpdir / '*'), '--shallow-check')
        assert isinstance(result.exception, TFRecordValidationError)

        with open(datadir / 'tfrecords' / 'simple' / 'validation-00000-of-00001', 'rb') as fr:
            with open(tmpdir / 'train-00000-of-00001', 'wb') as fw:
                fw.write(bytes(fr.read()))

        result = self._invoke_cli('validate', str(tmpdir / '*'), '--shallow-check')
        assert result.exit_code == 0
        assert result.exception is None

        result = self._invoke_cli('validate', str(tmpdir / '*'), '--deep-check')
        assert isinstance(result.exception, TFRecordValidationError)

        os.remove(tmpdir / '_tfrecord_metadata.json')
        result = self._invoke_cli('validate', str(tmpdir / '*'), '--shallow-check')
        assert isinstance(result.exception, TFRecordInfoMissingError)
