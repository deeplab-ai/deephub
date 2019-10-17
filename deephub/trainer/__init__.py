import logging
import os
from typing import Optional, List, Dict

import tensorflow as tf

from deephub.models import ModelBase
from deephub.models.feeders import FeederBase
from .metrics import export_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for deephub modules using Tensorflow pipeline

    Supports training models on CPU, GPU and Multi-GPU.

    """
    MIN_EVALUATION_PERIOD: float = 0.1

    def __init__(self, requested_gpu_devices: Optional[List[int]] = None):
        """
        Initialize trainer
        :param requested_gpu_devices: A list of device ids (in CUDA order) to use for training. If this
            is empty or None then it will fallback to CPU only training.

        """
        if not requested_gpu_devices:
            requested_gpu_devices = None
        self.requested_gpu_devices = requested_gpu_devices
        self.target_gpu_device_ids = self.apply_cuda_gpu_mask()

        self.number_of_devices = len(self.target_gpu_device_ids) if requested_gpu_devices is not None else 1

    @property
    def is_using_gpu(self) -> bool:
        """
        Check if this trainer will use GPU
        """
        return bool(self.requested_gpu_devices)

    def apply_cuda_gpu_mask(self) -> Optional[List[int]]:
        """
        Mask all CUDA devices that will not be used by this trainer

        :return: A list with new device ids after applying the mask
        """
        if self.requested_gpu_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, self.requested_gpu_devices))
            return list(range(0, len(self.requested_gpu_devices)))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            return None

    @property
    def distribution_strategy(self) -> tf.contrib.distribute.DistributionStrategy:
        """
        Construct a DistributionStrategy for running the model based on initialization parameters
        of trainer.
        """
        if not self.is_using_gpu:
            # Nothing is specified, fallback to CPU
            logger.info("Request training on local CPU.")
            return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")

        else:
            if len(self.target_gpu_device_ids) == 1:
                # Defined only one gpu
                logger.info(f"Request training on single local GPU with id {self.requested_gpu_devices[0]}.")
                return tf.contrib.distribute.OneDeviceStrategy(f"device:GPU:{self.target_gpu_device_ids[0]}")
            else:
                logger.info(f"Request training on {self.requested_gpu_devices} local GPUs using MirroredStrategy.")
                return tf.contrib.distribute.MirroredStrategy(
                    devices=[
                        f"device:GPU:{id_}"
                        for id_ in self.target_gpu_device_ids
                    ]
                )

    def train(self,
              model: ModelBase,
              train_feeder: FeederBase,
              epochs: int,
              save_summary_steps: int,
              save_checkpoint_steps: Optional[int] = None,
              save_checkpoint_secs: Optional[int] = None,
              eval_feeder: Optional[FeederBase] = None,
              validation_secs: Optional[int] = None,
              warm_start_check_point: Optional[str] = None,
              warm_start_variables_regex: Optional[str] = '.*',
              early_stopping_metric: Optional[str] = None,
              early_stopping_steps_without_decrease: Optional[int] = None,
              early_stopping_min_steps: Optional[int] = 1,
              early_stopping_hook_run_every_steps: Optional[int] = None) -> None:
        """
        Train a model given an existing train feeder and optionally an evaluation feeder.

        The training loop will run for as many epochs as defined or until all the input of
        train_feeder has been processed.

        Optionally the model can be evaluated multiple times between steps. The evaluation is controlled
        in temporal manner using the `eval_*_secs` arguments.
        Notice: Evaluation will start only after a checkpoint is written on disk.

        :param model: The model object to train
        :param train_feeder: A feeder with the training input dataset. Any implementation
        of FeederBase is accessible.
        :param epochs: The number of epochs to train a model. This parameter is forwarded to train_feeder
            to generate an ETL of that size.
        :param save_checkpoint_steps: Steps interval in order to save a new checkpoint.
        :param save_checkpoint_secs: Secs interval in order to save a new checkpoint.
        :param eval_feeder: If this is not None, training will also include evaluation in its loop.
            A feeder with the validation input dataset. Any implementation of FeederBase is accessible.
        :param validation_secs: Do not re-evaluate unless the last evaluation was started at least
            this many seconds ago. Of course, evaluation does not occur if no new checkpoints are
            available, hence, this is the minimum. Default value (None) means that every time a new checkpoint
            is saved, the evaluation step will be executed.
        :param save_summary_steps: Save summaries in events files every this many steps.
        :param warm_start_check_point: If set it will first warm start model variables with values
            loaded from an existing checkpoint. This can be either the directory with checkpoint
            file(s) (and the last will be used) or absolute path path to checkpoint file.
        :param warm_start_variables_regex: A regular expression (string) that captures which variables
            to warm-start (see tf.get_collection). This expression will only consider variables in
            the TRAINABLE_VARIABLES collection. By default all variables will be used.
        :param early_stopping_metric: String indicating the Metric for Early Stopping (e.g loss, accuracy)
                                      This parameter must be defined in order to enable the early stopping mechanism.
        :param early_stopping_steps_without_decrease: Number of steps without decrease in early_stopping_metric.
        :param early_stopping_min_steps: Number of minimum steps, in order to NOT execute early stopping even if the
                                         stopping criteria condition has been met.
        :param early_stopping_hook_run_every_steps: How often will the early stopping condition is checked. With the
                                                    default value (None) the hook will be executed once every epoch.
        """
        warm_start_settings = self._get_warm_start_settings(warm_start_check_point, warm_start_variables_regex)
        extra_run_config_params = self._get_extra_run_config_params(
            save_summary_steps, save_checkpoint_steps, save_checkpoint_secs, eval_feeder, validation_secs)

        if eval_feeder:
            validation_secs = extra_run_config_params.pop('validation_secs')

        estimator = self._get_estimator_from_model(model, warm_start_settings, extra_run_config_params)

        hooks = list()
        if early_stopping_metric is not None:
            hooks.append(tf.estimator.experimental.stop_if_no_decrease_hook(
                estimator,
                metric_name=early_stopping_metric,
                max_steps_without_decrease=early_stopping_steps_without_decrease,
                min_steps=early_stopping_min_steps,
                run_every_secs=None,
                run_every_steps=early_stopping_hook_run_every_steps)
            )

        max_steps = train_feeder.total_steps(epochs=epochs) / self.number_of_devices
        if max_steps is None:
            logger.warning("Training feeder does not report total_steps, training will be performed for ever or "
                           "until feeder raises an exception.")

        # Train and evaluation
        if eval_feeder:
            train_spec = tf.estimator.TrainSpec(
                input_fn=train_feeder.get_input_fn(epochs=epochs),
                max_steps=max_steps,
                hooks=hooks
            )

            eval_spec = tf.estimator.EvalSpec(
                input_fn=eval_feeder.get_input_fn(epochs=1),
                throttle_secs=validation_secs,
                steps=eval_feeder.total_steps(epochs=1)  # As evaluation is executed on 1 device (Only train_distribute
                                                         # has been declared in tf.contrib.distribute.DistributeConfig),
                                                         # se there is no need to divide with the number of devices.
            )

            tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
        # Just train
        else:
            estimator.train(
                input_fn=train_feeder.get_input_fn(epochs=epochs),
                max_steps=max_steps,
                hooks=hooks
            )

        self._apply_post_training_steps(model_dir=estimator.model_dir)

    @staticmethod
    def _get_extra_run_config_params(save_summary_steps, save_checkpoint_steps, save_checkpoint_secs,
                                     eval_feeder, validation_secs):
        # Only one of these variables can be specified
        if not bool(save_checkpoint_steps) ^ bool(save_checkpoint_secs):
            raise ValueError('You must specify one of save_checkpoint_steps or save_checkpoint_secs '
                             'but not both of them or none of them.')

        # If one of these has been specified, then a validation feeder should exists.
        if validation_secs:
            if not eval_feeder:
                raise ValueError('You have not specified a validation_feeder but you have '
                                 'set validation_secs variable.')

        if save_checkpoint_steps:
            # Checks only in case an eval feeder exists
            if eval_feeder:
                if validation_secs:
                    logger.warning(
                        f'You have specified to save checkpoints every {save_checkpoint_steps} steps and '
                        f'execute validation every {validation_secs} secs. WARNING: If you have specified to '
                        f'save checkpoints every 1000 steps and execute a validation every 3600 secs, and '
                        f'on the 1000th step you save a checkpoint. Say that the elapsed time since the last '
                        f'checkpoint is  3599 secs, then the evaluation will NOT be executed and you will have '
                        f'to wait for the next 1000 steps to happen in order to execute the validation step. '
                        f'This will happen for the next time too, so finally you have one validation every '
                        f'2000 steps or 7198 secs.')
                else:
                    # validation_secs has not been declared, so the validation will be executed every
                    # time a new checkpoint has been saved.
                    logger.warning(
                        f'You have not specified validation_secs variable, so the validation will be executed every '
                        f'save_checkpoint_steps({save_checkpoint_steps}) steps.')

                    # Set validation secs into a small value, so every time a checkpoint is saved the validation step
                    # will be executed
                    validation_secs = Trainer.MIN_EVALUATION_PERIOD

            save_checkpoints_dict = {'save_checkpoints_steps': save_checkpoint_steps}
        else:
            if eval_feeder:
                if not validation_secs:
                    validation_secs = min(Trainer.MIN_EVALUATION_PERIOD, save_checkpoint_secs)
                    logger.warning(f'You have not specified validation_secs variable, so the validation will be '
                                   f'executed every save_checkpoint_secs({save_checkpoint_secs}) secs.')

                else:
                    logger.warning(f'You have specified to save checkpoints every {save_checkpoint_secs} secs and '
                                   f'execute validation every {validation_secs} secs. WARNING: If you have '
                                   f'specified to save checkpoints every 3600 secs and execute a validation every '
                                   f'3601 secs, then when you save the first checkpoint on the 3600th sec the '
                                   f'validation step will NOT be executed because the interval for the evaluation '
                                   f'period has not passsed yet and as a result you have to wait for the next '
                                   f'checkpoint to be saved (after 3600 secs). In such a situation you are going to '
                                   f'execute one validation every 7200 secs.')

            save_checkpoints_dict = {'save_checkpoints_secs': save_checkpoint_secs}

        final_dict = {'save_summary_steps': save_summary_steps}
        if eval_feeder:
            final_dict.update({'validation_secs': validation_secs})

        final_dict.update(save_checkpoints_dict)

        return final_dict

    @staticmethod
    def _get_warm_start_settings(warm_start_check_point, warm_start_variables_regex):
        warm_start_settings = None
        if warm_start_check_point and warm_start_variables_regex is None:
            logger.error("For warm start to work you need to provide both checkpoint and regex")

        elif warm_start_check_point and warm_start_variables_regex:
            logger.info(f"Warm-starting model from checkpoint ")

            warm_start_settings = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=warm_start_check_point,
                vars_to_warm_start=warm_start_variables_regex
            )

        return warm_start_settings

    def _get_estimator_from_model(self, model: ModelBase,
                                  warm_start_settings: tf.estimator.WarmStartSettings,
                                  extra_run_config_params: Dict) -> tf.estimator.Estimator:
        """
        Get tf.Estimator from deephub.Model
        :param model: The original model to generate tf.Estimator from.
        :param warm_start_settings: Optional a WarmStartSettings object with warm start to initialize Estimator with.
        :param extra_run_config_params: Optional a dict with extra parameters to be used in tf.estimator.RunConfig
        :return: A fresh constructed tf.estimator.Estimator object
        """

        distribute_config = tf.contrib.distribute.DistributeConfig(
            train_distribute=self.distribution_strategy
        )

        run_config_params = {
            'experimental_distribute': distribute_config
        }
        run_config_params.update(extra_run_config_params)

        estimator = model.estimator(
            run_config_params=run_config_params,
            warm_start_settings=warm_start_settings
        )
        return estimator

    @staticmethod
    def _apply_post_training_steps(model_dir):
        """
        Applies post training actions

        """
        export_metrics(model_dir=model_dir)
