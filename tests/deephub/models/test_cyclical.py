import numpy as np
import tensorflow as tf
import math
from deephub.trainer.schedulers import cyclical_learning_rate


class TestCyclical:
    def test_exponential_clr(self):
        steps = np.arange(0, 100, 1)
        min_lr = 1e-8
        max_lr = 1e-3
        cycle_size_steps = 20
        expected_values = []
        actual_values = []
        with tf.Session() as sess:
            for global_step in steps:
                x = np.remainder(global_step, cycle_size_steps) / cycle_size_steps  # position in the cycle
                expected_clr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * x))

                tf_clr = cyclical_learning_rate(global_step=global_step,
                                                    min_lr=min_lr,
                                                    max_lr=max_lr,
                                                    cycle_size_steps=cycle_size_steps)
                expected_values.append(expected_clr)
                actual_values.append(sess.run(tf_clr))
        assert np.allclose(expected_values, actual_values)
