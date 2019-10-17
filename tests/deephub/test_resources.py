import os

import pytest
from pathlib import Path

from deephub.resources import get_resource_path, get_resources_writable_directory, ResourceNotFound
import deephub.resources


def create_user_fixture_files(user_dir):
    os.makedirs(os.path.join(user_dir, 'downloads'))
    os.makedirs(os.path.join(user_dir, 'data'))

    with open(os.path.join(user_dir, 'downloads', 'test_file1.json'), 'wt') as f:
        f.write('{ "foo": "bar" }')


def test_get_user_resources(monkeypatch, tmpdir):
    create_user_fixture_files(str(tmpdir))

    monkeypatch.setattr(deephub.resources, '_user_resources_dir', Path(str(tmpdir)))

    full_path = get_resource_path('downloads/test_file1.json')

    assert full_path.is_file()
    assert str(full_path).endswith('downloads/test_file1.json')
    assert str(full_path).startswith(str(tmpdir))


def test_notfound_resource():
    with pytest.raises(ResourceNotFound):
        get_resource_path('unknown.txt')


def test_get_existing_writable_directory(monkeypatch, tmpdir):
    monkeypatch.setattr(deephub.resources, '_user_resources_dir', Path(str(tmpdir)))

    full_path = get_resources_writable_directory('downloads')

    assert full_path.is_dir()
    assert str(full_path).startswith(str(tmpdir))


def test_get_non_existing_writable_directory(monkeypatch, tmpdir):
    monkeypatch.setattr(deephub.resources, '_user_resources_dir', Path(str(tmpdir)))

    assert not (Path(tmpdir) / 'new_one').is_dir()
    full_path = get_resources_writable_directory('new_one')

    assert full_path.is_dir()
    assert str(full_path).startswith(str(tmpdir))
