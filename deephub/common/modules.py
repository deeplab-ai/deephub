import importlib
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class ObjectNotFoundError(ImportError):
    """The specified object was not find in the package"""


class InstantiationError(RuntimeError):
    """An error occurred that prevented instantiating internal objects"""


def import_object(object_path: str, search_modules: Optional[List[str]] = None) -> Any:
    """
    Find and import a reference of an object, such as variables, classes, functions etc. It supports searching
    in multiple modules or import from an absolute module "path". The search will respect the order of the search
    modules.


    Examples: Get different references from a known module

    >>> # Get a reference to a class
    >>> MyClass = import_object("packagea.packageb:MyClass")

    >>> # Get a reference to a package
    >>> vara = import_object("packagea.moduleb:vara")

    Examples: Search for an object in multiple modules

    >>> # Search for a reference given a relative path
    >>> vara = import_object("moduleb:vara", ['packagea', 'packageb'])

    >>> varb = import_object("varb", ["packagea", "packageb"])

    :param object_path: The name of the object reference that we are search for.
        The name can be optionally prefixed with a dot-separated module path
        that describes the absolute location of the object. If `search_modules` is defined
        the path will be used as relative path inside each of the search modules. The path
        and object name are separated with the ":" colon character.
        The general format is:  [<package>.[<package>.]*:]object
    :param search_modules: A list of module defined in dot-separated format that will first try to
        import object from any of these relative paths. If the object is not found in any of these search modules,
         it will fallback to absolute path. This is expected to be a list of dot-separated module paths.
    :return: The imported object
    :raises ModuleNotFoundError: When the it couldn't find and import the module path of the object
    :raises ObjectNotFoundError: When the requested object was not found in the defined module path.
    :raises ValueError: If the object_path was not well defined.
    """

    if ':' not in object_path:
        if not search_modules:
            raise ValueError(f"Cannot search for object '{object_path}' because neither an absolute path "
                             f"was provided nor a list of modules to search for.")
        module_path = None
        object_name = object_path
    else:
        module_path, object_name = object_path.split(':')

    # Try to import as relative object in one of search modules
    if search_modules:
        for search_module in search_modules:
            absolute_path = ".".join([p for p in search_module.split('.') + [module_path] if p])
            absolute_path = ":".join([absolute_path, object_name])
            try:
                obj = import_object(absolute_path)
                logger.debug(f"Object '{absolute_path}' was successfully imported.")
                return obj
            except (ModuleNotFoundError, ObjectNotFoundError):
                continue

    if not module_path:
        # When path is not given, we do not perform absolute search
        logger.debug(f"Object '{object_path}' was not found.")
        raise ObjectNotFoundError(f"Object '{object_path}' was not found in any of the search modules.")

    # Perform absolute search
    module = importlib.import_module(module_path)
    if not hasattr(module, object_name):
        logger.debug(f"Object '{object_path}' was not found.")
        raise ObjectNotFoundError(f"Cannot find object '{object_name}' inside module '{module_path}'")

    return getattr(module, object_name)


def instantiate_from_dict(params: Dict[str, Any],
                          search_modules: Optional[List[str]] = None,
                          exclude_keys: Optional[List] = None,
                          class_name_key: str = 'class_type') -> Any:
    """
    Instantiate a class that was described in a dictionary format.

    The dictionary is expected to hold a class path as also all the arguments of the class constructor.

    Example instantiating a datetime object:

    >>> dt = instantiate_from_dict({'type': 'datetime:datetime', 'year': 2018, 'month': 10, 'day': 1})


    :param params: A map of parameters that will be used as keyword arguments for the class instantiation. A special
        entry with key `class_name_key` will be used to resolve the class reference.
        The rest of parameters except `exclude_keys` will be used as `kwargs`.
    :param search_modules: A list of extra modules to search and import class reference. See: `import_module`
    :param exclude_keys: A list of keys of params dictionary to exclude from kwargs that will be used for
        instantiation.
    :param class_name_key: The key in params that holds the class path.
    :return: The newly instantiated object
    :raise InstantiationError: When an error prevented from instantiating object.
    """
    if class_name_key not in params:
        raise InstantiationError(
            f"There is no entry with class type key '{class_name_key}' in configuration '{params}'")

    # Resolve class reference
    logger.debug(f"Try to import object '{params.get(class_name_key)}' for instantiation.")
    class_ref = import_object(params[class_name_key], search_modules=search_modules)

    # Construct kwargs
    if not exclude_keys:
        excluded_params = set()
    else:
        excluded_params = set(exclude_keys)
    excluded_params |= {class_name_key}

    class_params = {k: v
                    for k, v in params.items()
                    if k not in excluded_params}

    # Instantiate class
    try:
        logger.debug(f"Instantiating class '{class_ref.__name__}' with "
                     f"the following parmaters '{list(class_params.keys())}")
        return class_ref(**class_params)
    except Exception as e:
        raise InstantiationError(f"Error while instantiating class '{class_ref.__qualname__}': {e!s}")
