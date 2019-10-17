from abc import abstractmethod
from typing import Optional, Callable

import tensorflow as tf

InputFN = Callable[[], tf.data.Dataset]


class FeederBase:
    """
    Interface for implementing feeders

    Implementations should consider carefully the management of the steps and
    """

    @abstractmethod
    def get_input_fn(self, epochs: Optional[int] = None) -> InputFN:
        """
        Get the input_fn() function that will be used to construct the tf.data.Dataset

        :param epochs: The number of epochs that this pipeline will be used for.
        :return: A reference to an input_fn function
        """
        raise NotImplementedError()

    @abstractmethod
    def total_steps(self, epochs: int) -> Optional[int]:
        """
        The total number of steps that feeder can produce in one call of input_fn()

        :param epochs: The number of epochs that will be used for.
        :return: The number of steps or None if this feeder will produce steps for ever.
        """
        raise NotImplementedError()
