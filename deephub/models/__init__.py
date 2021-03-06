from typing import Callable, Optional, Dict
from abc import abstractmethod

import tensorflow as tf

from .feeders import FeederBase
from deephub.common.io import AnyPathType
import math

ModelFNType = Callable[[tf.Tensor, tf.Tensor, tf.estimator.ModeKeys, Dict], tf.estimator.EstimatorSpec]
EstimatorFN = Callable[[Optional[Dict]], tf.estimator.Estimator]


class ModelBase:
    """
    General model interface that can be used by trainer.
    To implement this interface you need to provide a way to translate your model in tf.estimator.Estimator type.
    """

    @abstractmethod
    def estimator(self,
                  run_config_params: Optional[Dict] = None,
                  warm_start_settings: Optional[tf.estimator.WarmStartSettings] = None) -> tf.estimator.Estimator:
        """
        Construct the tf.Estimator object of this model.
        :param run_config_params: Extract parameters to add at tf.estimator.RunConfig at instantiation of estimator
        :param warm_start_settings: Settings to configure warm start of model..
        """
        raise NotImplementedError()

    def predict(self, pred_feed: FeederBase):
        return self.estimator().predict(pred_feed.get_input_fn(1))


class EstimatorModel(ModelBase):
    """
    Model implementation for native tf.Estimator objects
    """

    def __init__(self, model_dir: AnyPathType, **model_params: Optional[Dict]) -> None:
        """
        Instantiate a new estimator model.

        :param model_dir: The directory to read and write checkpoints
        """
        self.model_dir = model_dir
        self.model_params = model_params

    def estimator(self,
                  run_config_params: Optional[Dict] = None,
                  warm_start_settings: Optional[tf.estimator.WarmStartSettings] = None) -> tf.estimator.Estimator:
        if run_config_params is None:
            run_config_params = {}

        # Apply default parameters to run config
        default_run_config_params = {
            'model_dir': self.model_dir
        }
        default_run_config_params.update(run_config_params)

        run_config = tf.estimator.RunConfig(**default_run_config_params)

        return tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=run_config,
            params=self.model_params,
            warm_start_from=warm_start_settings
        )

    def model_fn(self, features, labels, mode, params, config) -> tf.estimator.EstimatorSpec:
        raise NotImplementedError()


class KerasModel(ModelBase):
    """
    Model implementation for Keras model
    """

    def __init__(self, model_dir: AnyPathType):
        """
        Instantiate a new Keras model.

        :param model_dir: The directory to read and write checkpoints
        """
        self.model_dir = model_dir

    def estimator(self,
                  run_config_params: Optional[Dict] = None,
                  warm_start_settings: Optional[tf.estimator.WarmStartSettings] = None) -> tf.estimator.Estimator:
        if run_config_params is None:
            run_config_params = {}

        run_config = tf.estimator.RunConfig(
            model_dir=self.model_dir,
            save_summary_steps=30,
            **run_config_params
        )

        return tf.keras.estimator.model_to_estimator(
            self.keras_model_fn(),
            model_dir=self.model_dir,
            config=run_config,
        )

    @abstractmethod
    def keras_model_fn(self):
        raise NotImplementedError()
