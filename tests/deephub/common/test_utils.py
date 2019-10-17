from __future__ import absolute_import

from deephub.common.utils import size_formatted, merge_dictionaries, deep_update_dict


def test_size_formatted():
    assert size_formatted(0) == '  0B'

    assert size_formatted(10) == ' 10B'

    assert size_formatted(1024) == '1.0KiB'

    assert size_formatted(2000) == '2.0KiB'

    assert size_formatted(1548576) == '1.5MiB'


def test_merge_dictionaries():
    _in1 = {
        'a': 1,
        'b': [1, 2]
    }

    _in2 = {
        'a': 3,
        'c': 5,
        'd': 'bar'
    }

    _in3 = {
        'b': 3,
        'c': 6,
        'f': 7
    }

    # Test merging the same
    assert merge_dictionaries({'a': 1}) == {'a': 1}
    assert merge_dictionaries(_in1) == _in1, "Merging one dictionary must return an equal dictionary"
    assert merge_dictionaries(_in1) is not _in1, "Merging must always create a new dictionary"

    assert merge_dictionaries(_in1, _in2) == {
        'a': 3,
        'b': [1, 2],
        'c': 5,
        'd': 'bar'
    }

    assert merge_dictionaries(_in2, _in1) == {
        'a': 1,
        'b': [1, 2],
        'c': 5,
        'd': 'bar'
    }, "Merging in opposite order should give different results"

    assert merge_dictionaries(_in1, _in2, _in3) == {
        'a': 3,
        'b': 3,
        'c': 6,
        'd': 'bar',
        'f': 7
    }


class TestDeepUpdateDict:

    def test_simple_deep_update_dict(self):
        res = deep_update_dict(
            {
                'a': 1,
                'b': []
            },
            {
                'a': ['something', 'else'],
                'c': {
                    'a': 'new'
                }
            })
        assert res == {
            'a': ['something', 'else'],
            'b': [],
            'c': {
                'a': 'new'
            }
        }

    def test_deep_update_dict_deeper(self):
        res = deep_update_dict(
            {
                'a': 1,
                'b': {
                    'here': 2,
                    'another': {},
                    'yetanother': {
                        'level3_0': 3,
                        'level3_1': 5
                    }
                }
            },
            {
                'c': 32,
                'b': {
                    'here': 'new',
                    'yetanother': {
                        'level3_1': 'updated',
                        'level3_2': 4
                    }
                }
            })
        assert res == {
            'a': 1,
            'b': {
                'here': 'new',
                'another': {},
                'yetanother': {
                    'level3_0': 3,
                    'level3_1': 'updated',
                    'level3_2': 4
                }
            },
            'c': 32
        }
