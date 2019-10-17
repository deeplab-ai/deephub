import logging
import os

from .definition import VariantDefinition
from deephub.resources import get_resource_path
import metayaml

logger = logging.getLogger(__name__)


class VariantDefinitionError(RuntimeError):
    """An error at processing a variant definition"""
    pass


class UnknownVariant(VariantDefinitionError):
    """An unknown variant was requested"""
    pass


def load_variant(variant_name: str, variants_dir: str) -> VariantDefinition:
    if variants_dir is None:
        variants_dir = get_resource_path("config/variants")
    subpaths = [variants_dir, variant_name+".yaml"]
    yaml_fname = os.path.join(*subpaths)
    if os.path.exists(yaml_fname):
        definition = metayaml.read(yaml_fname)
    else:
        raise UnknownVariant(f"Cannot find product with name '{variant_name}'")

    return VariantDefinition(variant_name, definition)
