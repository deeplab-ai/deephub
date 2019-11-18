import tensorflow as tf
import math


def cyclical_learning_rate(global_step: int, min_lr: float, max_lr: float, cycle_size_steps: int) -> tf.float32:
    """
    Cyclical learning rate (CLR) implementation.
    :param global_step: Tensorflow Global step
    :param min_lr: minimum learning rate of the cycle
    :param max_lr: maximum learning rate of the cycle
    :param cycle_size_steps: steps for the cycle
    :return: The calculated learning rate
    """
    if global_step is None:
        raise ValueError("global_step is required for cyclical_learning_rate.")
    if max_lr <= min_lr:
        raise ValueError("min_lr must be smaller than max_lr.")
    with tf.name_scope("CyclicalLearningRate"):
        min_lr = tf.convert_to_tensor(min_lr, name="learning_rate")
        global_step = tf.cast(global_step, tf.float32)
        cycle_size_steps = tf.cast(cycle_size_steps, tf.float32)
        scaled_position = tf.floormod(x=global_step, y=cycle_size_steps)
        x = tf.truediv(x=scaled_position, y=tf.cast(cycle_size_steps, dtype=tf.float32))
        return min_lr + 0.5 * (1 + tf.math.cos(math.pi * x)) * (max_lr - min_lr)