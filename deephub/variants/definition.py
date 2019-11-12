import logging

import os
import yaml

from typing import Dict, Any, List, Optional

from deephub.common.modules import instantiate_from_dict
from deephub.common.utils import deep_update_dict
from deephub.models import ModelBase
from deephub.models.feeders import FeederBase
from deephub.trainer import Trainer

logger = logging.getLogger(__name__)


class VariantDefinition:

    def __init__(self, name: str, definition: Dict) -> None:
        """
        Initialize a variant definition object

        :param name: The name of the variant
        :param definition: The raw configuration of the variant in nested dictionary format.
        """
        if not isinstance(definition, dict):
            raise TypeError(f"The definition must a nested dictionary object, instead a {type(definition)} was given.")
        self.name = name
        self.definition = definition

    @staticmethod
    def dict_from_dot_format(param_path: str, value: Any) -> Dict:
        """
        Get a nested dictionary with all the keys in param path and the value assigned to leaf key.

        Example:

        >>> VariantDefinition.dict_from_dot_format('foo.bar.slack', 1)
        >>> {'foo':{
        >>>     'bar':{
        >>>         'slack': 1
        >>>         }
        >>>     }
        >>> }

        :param param_path: A dot-separated hierarchical path of the parameter key.
        :param value: The value to assign on this path.
        :return: The constructed dictionary
        """
        constructed_dict = value
        for key in reversed(param_path.split('.')):
            if not key:
                continue
            constructed_dict = {
                key: constructed_dict
            }
        return constructed_dict

    def get(self, param_path: str) -> Any:
        """
        Get the raw value of a parameter in configuration.

        :param param_path: The dot-separated hierarchical path of the parameter key.
        :return: The raw value of the parameter
        :raises KeyError: When paths are not found in configuration
        """
        current_dict = self.definition
        for key in param_path.split('.'):
            current_dict = current_dict[key]

        return current_dict

    def has(self, param_path: str) -> Any:
        """
        Check that parameter exist in variant definition

        :param param_path: The dot-separated hierarchical path of the parameter key.
        :return: True if found else False
        """

        current_dict = self.definition
        for key in param_path.split('.'):
            if not isinstance(current_dict, dict) or key not in current_dict:
                return False
            current_dict = current_dict[key]
        return True

    def set(self, param_path: str, value: Any) -> None:
        """
        Change the value of a definition parameter. The operation will create any missing nodes and it will
        extend existing ones.

        :param param_path: The dot-separated hierarchical path of the parameter key.
        :param value: The new value of the parameter.
        """
        updated_node = self.dict_from_dot_format(param_path, value)
        deep_update_dict(self.definition, updated_node)

    def sub_get(self, param_path: str, exclude_keys: Optional[List[str]] = None) -> Dict:
        """
        Get a sub tree of a configuration node, excluding some of its children.

        :param param_path: The dot-separated hierarchical path of the parameter key.
        :param exclude_keys: A list of children keys that must be removed from the final
            configuration group.
        :return: A dictionary with all key-value pairs of this parameter path without the excluded.
        """
        current_dict = self.definition
        for key in param_path.split('.'):
            current_dict = current_dict[key]

        if not isinstance(current_dict, dict):
            raise TypeError(f"Parameter '{param_path}' must be dictionary to support sub_get operation.")

        if not exclude_keys:
            return current_dict

        return {
            k: v
            for k, v in current_dict.items()
            if k not in set(exclude_keys)
        }

    def create_model(self) -> ModelBase:
        """
        Create a fresh model instance based on the definition of the variant

        :return: The instantiated Model object
        """
        model = instantiate_from_dict(self.get('model'),
                                      search_modules=[self.get('model.module_path')])
        if not isinstance(model, ModelBase):
            logger.warn(f"Class '{model.__class__.__name__}' that is used as model type is not subclass of ModelBase.")
        return model

    def create_feeder(self, feeder_config_path: str) -> FeederBase:
        """
        Create a fresh Feeder object instance based on the definition of the variant.

        :param feeder_config_path: A dot-separated path in the variant definition
            that holds the feeder configuration. Usually a value of `train.train_feeder` or
            `train.eval_feeder`

        :return: The instantiated Feeder object
        """
        if self.has('train.train_feeder'):
            search_modules = [self.get('train.train_feeder.module_path')]
        elif self.has('train.eval_feeder'):
            search_modules = [self.get('train.eval_feeder.module_path')]
        elif self.has('predict.predict_feeder'):
            search_modules = [self.get('predict.predict_feeder')]
        else:
            raise ValueError('Variant must have one of the following: \n ' +
                             'train.train_feeder, train.eval_feeder, predict.predict_feeder')

        feeder = instantiate_from_dict(
            self.get(feeder_config_path),
            search_modules=search_modules,
            exclude_keys=['model_dir', 'module_path', 'class_type']
        )

        if not isinstance(feeder, FeederBase):
            logger.warn(f"Class '{feeder.__class__.__name__}' that is used as feeder "
                        f"type is not subclass of FeederBase.")
        return feeder

    def train(self, trainer: Trainer) -> ModelBase:
        """
        Instantiate this variant's components and train based on definition

        :param trainer: The trainer object to use for training.
        :return: The trained model
        """

        logger.info(f"Requested training variant '{self.name}' with definition: {self.definition}")

        model = self.create_model()

        # Create feeders
        train_feeder = self.create_feeder('train.train_feeder')
        if self.has('train.eval_feeder'):
            eval_feeder = self.create_feeder('train.eval_feeder')
        else:
            eval_feeder = None

        # Dump model and train params from variants yaml
        self._dump_variant_definition()

        # Invoke training
        train_params = self.sub_get('train', exclude_keys=['train_feeder', 'eval_feeder'])
        trainer.train(model=model, train_feeder=train_feeder, eval_feeder=eval_feeder, **train_params)

        return model

    def _dump_variant_definition(self):
        if not os.path.exists(self.definition['model']['model_dir']):
            os.mkdir(self.definition['model']['model_dir'])

        with open(os.path.join(self.definition['model']['model_dir'], 'experiment_params.yaml'), '+w') as f:
            yaml.dump(self.definition, f)
