import pytest

from deephub.common.gs_io import parse_gs_url


def test_parse_gs_url():
    bucket, path = parse_gs_url('gs://bucket1/path')
    assert bucket == 'bucket1'
    assert path == 'path'

    bucket, path = parse_gs_url('gs://some-bucket/a/very/long/path/with.extension.zip')
    assert bucket == 'some-bucket'
    assert path == 'a/very/long/path/with.extension.zip'


def test_pare_non_gs_url():
    with pytest.raises(ValueError):
        parse_gs_url('/simple/path')

    with pytest.raises(ValueError):
        parse_gs_url('http://server/simple/path')

    with pytest.raises(ValueError):
        parse_gs_url('http://another:12/simple/path')
