from typing import Callable


def constant_schedule(learning_rate: float) -> Callable[[float], float]:
    def func(progress_remaining: float):
        return learning_rate

    return func


def linear_schedule(initial_learning_rate: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_learning_rate

    return func
