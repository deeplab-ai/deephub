from unittest import mock

import pytest
import yaml

from deephub.models import FeederBase
from deephub.variants.definition import VariantDefinition


class TestVariantDefinition:
    VARIANT_DEF_A = {
        'model': {
            'module_path': 'somePath',
            'class_type': 'something',
            'foo': '1'
        },
        'train': {
            'epochs': 100,
            'train_feeder': {
                'module_path': 'somePathFeeder',
                'class_type': 'SomeFeedder',
                'files': 'something'
            }
        }
    }


    def test_dict_from_dot_format(self):
        assert VariantDefinition.dict_from_dot_format('foo', 1) == {'foo': 1}

        assert VariantDefinition.dict_from_dot_format('foo.bar', 1) == {
            'foo': {
                'bar': 1
            }
        }

        assert VariantDefinition.dict_from_dot_format('foo.bar.', 'a string') == {
            'foo': {
                'bar': 'a string'
            }
        }

        assert VariantDefinition.dict_from_dot_format('.with.a.complex', {'another'}) == {
            'with': {
                'a': {
                    'complex': {'another'}
                }
            }
        }

    def test_construction(self):
        with pytest.raises(TypeError):
            VariantDefinition('name', None)

        with pytest.raises(TypeError):
            VariantDefinition('name', ['something'])

        prod = VariantDefinition(
            'the name',
            {
                'foo': {
                    'bar': 1
                }})

        assert prod.name == 'the name'
        assert prod.definition == {
            'foo': {
                'bar': 1
            }}

    def test_get(self):
        prod = VariantDefinition(
            'variant',
            {
                'foo': {
                    'bar': 1
                }})

        assert prod.get('foo.bar') == 1

        with pytest.raises(KeyError):
            prod.get('foo.found')

    def test_has(self):
        prod = VariantDefinition(
            'variant',
            {
                'foo': {
                    'bar': {
                        'a': 1,
                        'b': 2
                    },
                    'another': 15
                }})

        assert prod.has('foo')
        assert prod.has('foo.bar')
        assert prod.has('foo.bar.a')
        assert prod.has('foo.another')
        assert not prod.has('unknown')
        assert not prod.has('foo.unknown')
        assert not prod.has('foo.bar.unknown')
        assert not prod.has('foo.bar.a.unknown')
        assert not prod.has('foo.another.unknown')

    def test_simple_sub_get(self):
        prod = VariantDefinition(
            'variant',
            {
                'foo': {
                    'bar': {
                        'a': 1,
                        'b': 2
                    },
                    'another': 15
                }})

        assert prod.sub_get('foo') == {
            'bar': {
                'a': 1,
                'b': 2
            },
            'another': 15
        }

        assert prod.sub_get('foo.bar') == {
            'a': 1,
            'b': 2
        }

    def test_sub_get_with_excluded(self):
        prod = VariantDefinition(
            'variant',
            {
                'foo': {
                    'bar': {
                        'a': 1,
                        'b': 2
                    },
                    'another': 15
                }})

        assert prod.sub_get('foo', exclude_keys=['unknown']) == {
            'bar': {
                'a': 1,
                'b': 2
            },
            'another': 15
        }
        assert prod.sub_get('foo', exclude_keys=['unknown', 'another', 'a']) == {
            'bar': {
                'a': 1,
                'b': 2
            }
        }

        assert prod.sub_get('foo.bar', exclude_keys=['unknown', 'b']) == {
            'a': 1
        }

    def test_sub_get_wrong_type(self):
        prod = VariantDefinition(
            'variant',
            {
                'foo': {
                    'bar': {
                        'a': 1,
                        'b': 2
                    },
                    'another': 15
                }})

        with pytest.raises(TypeError):
            prod.sub_get('foo.another')

    def test_sub_get_unknown(self):
        prod = VariantDefinition(
            'variant',
            {
                'foo': {
                    'bar': {
                        'a': 1,
                        'b': 2
                    },
                    'another': 15
                }})

        with pytest.raises(KeyError):
            prod.sub_get('foo.unknown')

    def test_create_model(self, tmpdir):
        with mock.patch('deephub.variants.definition.instantiate_from_dict') as mocked_instantiated:
            self.VARIANT_DEF_A['model']['model_dir'] = str(tmpdir / 'VARIANT_DEF_A')
            prod_a = VariantDefinition(
                'varianta',
                self.VARIANT_DEF_A
            )

            model = prod_a.create_model()

            mocked_instantiated.assert_called_with({
                'module_path': 'somePath',
                'class_type': 'something',
                'foo': '1',
                'model_dir': str(tmpdir / 'VARIANT_DEF_A')
            },
            search_modules=['somePath'])

    def test_create_feeder(self, tmpdir):
        self.VARIANT_DEF_A['model']['model_dir'] = str(tmpdir / 'VARIANT_DEF_A')
        prod_a = VariantDefinition(
            'varianta',
            self.VARIANT_DEF_A
        )

        with mock.patch('deephub.variants.definition.instantiate_from_dict') as mocked_instantiated:
            feeder = prod_a.create_feeder('train.train_feeder')

            mocked_instantiated.assert_called_with(
                {
                    'module_path': 'somePathFeeder',
                    'class_type': 'SomeFeedder',
                    'files': 'something'
                },
                exclude_keys=['model_dir', 'module_path', 'class_type'],
                search_modules=['somePathFeeder'])

    def test_create_feeder_unknwon(self, tmpdir):
        self.VARIANT_DEF_A['model']['model_dir'] = str(tmpdir / 'VARIANT_DEF_A')
        prod_a = VariantDefinition(
            'varianta',
            self.VARIANT_DEF_A
        )

        with pytest.raises(KeyError):
            feeder = prod_a.create_feeder('train.unknwon')

    def test_export_to_yaml(self, tmpdir):
        variant_def = {
            'model': {
                'module_path': 'deephub.models.registry.toy',
                'class_type': 'DebugToyModel',
                'model_dir': str(tmpdir / 'test_exported_yaml')
            },
            'train': {
                'epochs': 1,
                'train_feeder': {
                    'module_path': 'deephub.models.feeders',
                    'class_type': 'MemorySamplesFeeder',
                    'x': {'features': [[1, 1], [1, 1]]},
                    'y': {'labels': [1, 1]}
                },
                'eval_feeder': {
                    'module_path': 'deephub.models.feeders',
                    'class_type': 'MemorySamplesFeeder',
                    'x': {'features': [[1, 1], [1, 1]]},
                    'y': {'labels': [1, 1]}
                }
            }
        }

        prod_b = VariantDefinition(
            'variant', variant_def
        )

        trainer = mock.MagicMock()
        prod_b.train(trainer)

        with open(str(tmpdir / 'test_exported_yaml/experiment_params.yaml'), 'r') as f:
            data = yaml.load(f)

        assert data == variant_def and data is not variant_def
