import os

import pytest
from pathlib import Path

from deephub.common.io import (append_to_json_file, append_to_txt_file,
                                     load_json_object, file_md5, resolve_glob_pattern)


def get_content(fname):
    with open(str(fname), 'r') as f:
        return f.read()


def touch(path):
    directory_path = Path(path).parent
    if not directory_path.exists():
        os.makedirs(directory_path)
    with open(path, 'w+'):
        pass


def test_append_to_txt_file(tmpdir):
    fname1 = os.path.join(str(tmpdir), 'file1.json')

    append_to_txt_file(fname1, "some text")
    assert get_content(fname1) == "some text"

    # Append something more
    append_to_txt_file(fname1, "lorium ipsium")
    assert get_content(fname1) == "some textlorium ipsium"

    # filename is of Path type
    append_to_txt_file(Path(fname1), " else\n")
    assert get_content(fname1) == "some textlorium ipsium else\n"


def test_append_to_json_file(tmpdir):
    fname1 = os.path.join(str(tmpdir), 'file1.json')

    append_to_json_file(fname1, {"a": 123, "b": "text"})
    assert get_content(fname1) == '{"a":123,"b":"text"}\n'

    append_to_json_file(fname1, ["a", "b", "c"])
    assert get_content(fname1) == '{"a":123,"b":"text"}\n["a","b","c"]\n'

    # filename is of Path type
    append_to_json_file(Path(fname1), ["a", "b", "c"])
    assert get_content(fname1) == '{"a":123,"b":"text"}\n["a","b","c"]\n["a","b","c"]\n'


def test_load_json_object(tmpdir):
    fname1 = os.path.join(str(tmpdir), 'file1.json')

    with open(fname1, "wt") as f:
        f.write('{"a":123,"b":"text"}\n')

    obj = load_json_object(fname1)
    assert obj == {"a": 123, "b": "text"}

    # filename is of Path type
    obj = load_json_object(Path(fname1))
    assert obj == {"a": 123, "b": "text"}


def test_file_md5_non_existing(tmpdir):
    with pytest.raises(IOError):
        file_md5(tmpdir / 'unknown_file')


def test_file_md5(datadir):
    hash = file_md5(datadir / 'random_400k.md5')
    assert hash == 'c10bd83ed1617fd2d9969f949d12a9b2'


def test_resolve_glob_pattern_folders_match(tmpdir):
    touch(tmpdir / "root.info")
    touch(tmpdir / "folderA" / "a.txt")
    touch(tmpdir / "folderB" / "c.info")
    touch(tmpdir / "folderB" / "C" / "c.txt")

    # 1st level with and without folders
    matches = resolve_glob_pattern('folder*', tmpdir, match_folders=False)
    assert matches == []

    matches = resolve_glob_pattern('folder*', tmpdir, match_folders=True)
    assert sorted(matches) == [
        tmpdir / 'folderA',
        tmpdir / 'folderB'
    ]

    # All levels with and without folders
    matches = resolve_glob_pattern('**/*', tmpdir, match_folders=True)
    assert sorted(matches) == [
        tmpdir / 'folderA',
        tmpdir / 'folderA' / 'a.txt',
        tmpdir / 'folderB',
        tmpdir / 'folderB' / 'C',
        tmpdir / 'folderB' / 'C' / 'c.txt',
        tmpdir / 'folderB' / 'c.info',
        tmpdir / 'root.info'
    ]

    matches = resolve_glob_pattern('**/*', tmpdir, match_folders=False)
    assert sorted(matches) == [
        tmpdir / 'folderA' / 'a.txt',
        tmpdir / 'folderB' / 'C' / 'c.txt',
        tmpdir / 'folderB' / 'c.info',
        tmpdir / 'root.info'
    ]


def test_resolve_glob_pattern(tmpdir):
    # Prepare a test structure
    touch(tmpdir / "root.txt")
    touch(tmpdir / "root.info")
    touch(tmpdir / "folderA" / "a.txt")
    touch(tmpdir / "folderA" / "b.txt")
    touch(tmpdir / "folderA" / "b.info")
    touch(tmpdir / "folderB" / "b.txt")
    touch(tmpdir / "folderB" / "c.txt")
    touch(tmpdir / "folderB" / "c.info")
    touch(tmpdir / "folderB" / "C" / "c.txt")
    touch(tmpdir / "folderB" / "C" / "c.info")

    root_text = resolve_glob_pattern("*.txt", tmpdir)
    assert sorted(root_text) == sorted([
        tmpdir / 'root.txt'
    ])

    all_text = resolve_glob_pattern("**/*.txt", tmpdir)
    assert sorted(all_text) == sorted([
        tmpdir / 'root.txt',
        tmpdir / 'folderA' / 'a.txt',
        tmpdir / 'folderA' / 'b.txt',
        tmpdir / 'folderB' / 'b.txt',
        tmpdir / 'folderB' / 'c.txt',
        tmpdir / 'folderB' / 'C' / 'c.txt',

    ])

    # Check on current working directory
    os.chdir(tmpdir / 'folderA')
    folder_a_info = resolve_glob_pattern("*.info")
    assert sorted(folder_a_info) == sorted([
        tmpdir / 'folderA' / 'b.info',
    ])

    # glob with absolute pattern
    folder_b_recurse_info = resolve_glob_pattern(str(tmpdir / 'folderB' / "**/*.info"))
    assert sorted(folder_b_recurse_info) == sorted([
        tmpdir / 'folderB' / 'c.info',
        tmpdir / 'folderB' / 'C' / 'c.info',
    ])

    # raise when starting directory and absolute path have been set
    with pytest.raises(ValueError):
        resolve_glob_pattern(str(tmpdir / '**/*.txt'), tmpdir)
