import inspect
import pytest
import datetime

from deephub.common.modules import import_object, ObjectNotFoundError, instantiate_from_dict, InstantiationError


class TestImportObject:
    def test_absolute_importing_standard_classes(self):
        path_class = import_object('pathlib:Path')
        assert inspect.isclass(path_class)
        assert path_class.__name__ == 'Path'

    def test_absolute_importing_local_class(self):
        not_found_exc = import_object('deephub.resources:ResourceNotFound')

        assert inspect.isclass(not_found_exc)
        assert issubclass(not_found_exc, Exception)
        assert not_found_exc.__name__ == 'ResourceNotFound'

    def test_import_unknown_module(self):
        with pytest.raises(ModuleNotFoundError):
            import_object('deephub.unkown_module:ResourceNotFound')

    def test_import_unknown_object(self):
        with pytest.raises(ObjectNotFoundError):
            import_object('deephub:WrongReference')

    def test_search_import_one_path(self):
        with pytest.raises(ModuleNotFoundError):
            import_object('resources:ResourceNotFound')

        not_found_exc = import_object('resources:ResourceNotFound', ['deephub'])

        assert not_found_exc.__name__ == 'ResourceNotFound'

    def test_search_import_multi_path(self):
        with pytest.raises(ModuleNotFoundError):
            import_object('resources:ResourceNotFound')

        not_found_exc = import_object('resources:ResourceNotFound', ['os', 'deephub'])

        assert not_found_exc.__name__ == 'ResourceNotFound'

    def test_false_search_import(self):
        """It will use search path but the path is given in absolute format"""

        not_found_exc = import_object('deephub.resources:ResourceNotFound', ['os', 'sys'])

        assert not_found_exc.__name__ == 'ResourceNotFound'

    def test_search_wrong_format(self):
        """It will use search path but the path is given in absolute format"""

        with pytest.raises(ValueError):
            import_object('ResourceNotFound')

    def test_search_object_without_path(self):
        not_found_exc = import_object('ResourceNotFound', ['os', 'deephub.resources'])

        assert not_found_exc.__name__ == 'ResourceNotFound'


class TestInstantiateFromDict:

    def test_simple_scenario(self):
        dt = instantiate_from_dict({
            'class_type': 'datetime:datetime',
            'year': 2018,
            'month': 9,
            'day': 8,
            'hour': 7,
            'minute': 6,
            'second': 5
        })
        assert datetime.datetime(2018, 9, 8, 7, 6, 5) == dt

    def test_search_modules(self):
        with pytest.raises(ValueError):
            instantiate_from_dict(
                {
                    'class_type': 'datetime',
                    'year': 2018,
                    'month': 9,
                    'day': 8,
                })

        with pytest.raises(ObjectNotFoundError):
            instantiate_from_dict(
                {
                    'class_type': 'datetime',
                    'year': 2018,
                    'month': 9,
                    'day': 8,
                },
                search_modules=['os'])

        dt = instantiate_from_dict(
            {
                'class_type': 'datetime',
                'year': 2018,
                'month': 9,
                'day': 8,
            },
            search_modules=['datetime'])

        assert dt == datetime.datetime(2018, 9, 8)

    def test_excluded_params(self):
        with pytest.raises(InstantiationError):
            instantiate_from_dict(
                {
                    'class_type': 'datetime:datetime',
                    'year': 2018,
                    'month': 9,
                    'day': 8,
                    'wrong': 15
                })

        instantiate_from_dict(
            {
                'class_type': 'datetime:datetime',
                'year': 2018,
                'month': 9,
                'day': 8,
                'wrong': 15
            },
            exclude_keys=['wrong'])

    def test_class_name_key(self):
        with pytest.raises(InstantiationError):
            instantiate_from_dict(
                {
                    'wrong_type': 'datetime:datetime',
                    'year': 2018,
                    'month': 9,
                    'day': 8,
                })

        dt = instantiate_from_dict(
            {
                '_class_': 'datetime:datetime',
                'year': 2018,
                'month': 9,
                'day': 8,
            },
            class_name_key='_class_')

        assert dt == datetime.datetime(2018, 9, 8)
