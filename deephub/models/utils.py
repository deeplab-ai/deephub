from typing import List, Optional
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class NoTrainableVariablesError(RuntimeError):
    """Error raised when all trainable variable variables filtered out"""
    pass


def get_fine_tuning_var_list(trainable_scope_prefixes: Optional[str]) -> Optional[List[tf.Variable]]:
    """
    Filter the trainable variables that optimizer should update to perform fine-tuning on
    a pre trained model.

    This function is expected to be used to create_the `var_list` for the
    minimization function from string that was given through model_fn `params`. e.g.

    >>> optimizer.minimize(
    >>>     loss_op,
    >>>     var_list=get_fine_tuning_var_list(params.get('trainable_scopes')))

    :param trainable_scope_prefixes: A comma separated list of variable scope prefixes that
        variables must have to keep them for training
    :return: The final list of tf.Variable that matched with `trainable_scope_prefixes` filter. Or
    None if no filter was requested.
    """

    if not trainable_scope_prefixes:
        return None

    scope_prefixes = [scope
                      for scope in map(str.strip, trainable_scope_prefixes.split(','))
                      if scope]

    logger.info(f"Fine-tuning has been enabled for scopes prefixes: {scope_prefixes}")

    var_list = [v for v in tf.trainable_variables()
                for scope in scope_prefixes
                if v.name.startswith(scope)]

    if not var_list:
        logger.error("No trainable variable was matched for fine-tuning with selected scope prefixes")
        raise NoTrainableVariablesError(f"No trainable variable was matched for fine-tuning with "
                                        f"selected scope prefixes: {scope_prefixes}")

    if len(var_list) == len(tf.trainable_variables()):
        logger.warning("Fine-tuning is effectively disabled as scope rules matched ALL trainable variables.")

    return var_list


def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
    """Returns a piecewise learning rate tensor.
  Args:
    piecewise_learning_rate_schedule: The --piecewise_learning_rate_schedule
      parameter
    global_step: Scalar tensor representing the global step.
    num_batches_per_epoch: float indicating the number of batches per epoch.
  Returns:
    A scalar float tensor, representing the learning rate.
  Raises:
    ValueError: piecewise_learning_rate_schedule is not formatted correctly.
  """

    pieces = piecewise_learning_rate_schedule.split(':')
    if len(pieces) % 2 == 0:
        raise ValueError('--piecewise_learning_rate_schedule must have an odd '
                         'number of components')
    values = []
    boundaries = []
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            try:
                values.append(float(piece))
            except ValueError:
                raise ValueError('Invalid learning rate: ' + piece)
        else:
            try:
                boundaries.append(int(int(piece) * num_batches_per_epoch) - 1)
            except ValueError:
                raise ValueError('Invalid epoch: ' + piece)
    return tf.train.piecewise_constant(global_step, boundaries, values,
                                       name='piecewise_learning_rate')
