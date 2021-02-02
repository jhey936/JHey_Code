# coding=utf-8
"""Provides some quality of life utilities to make testing easier"""
import numpy as np


# solves issues elsewhere to just accept *args and **kwargs
# noinspection PyUnusedLocal
def set_numpy_seed(*args, **kwargs) -> None:
    """Sets the numpy random seed to a known value for testing distributions to ensure consistency"""
    np.random.seed(123)


def parse_info_log(f_n: str="info.log", size=150) -> bytes:
    """parses the log and returns the last entry"""

    with open(f_n, "rb") as f:
        f.seek(-size, 2)
        data = f.read()
        return data


def get_dummy_population(minimum: float or int=-10.0, maximum: float or int=-0.1,
                         members: int=10) -> list:
    """Creates a dummy population for testing"""
    return [DummyPopMember(f) for f in np.linspace(minimum, maximum, members)]


class DummyPopMember(object):
    """Dummy member of a population for testing"""
    def __init__(self, cost) -> None:
        self.cost = cost

    def __repr__(self) -> str:
        return "Cost:{}".format(self.cost)


def check_list_almost_equal(array1: list or np.array or set or tuple,
                            array2: list or np.array or set or tuple,
                            decimal: int=7, log=None) -> bool:
    """Utility to check if two array_like objects are almost equal

    Args:
        array1: list, array, set or tuple
        array2: list, array, set or tuple
        decimal: max acceptable error
        log: logging.Logger instance if logging is required (it probably isn't)

    Returns:
        bool, True if lists are almost the same,
              False if they differ before decimal in any position

    """

    try:
        np.testing.assert_array_almost_equal(array1, array2, decimal=decimal)
        return True
    except AssertionError:
        if log is not None:
            log.debug("{} !~ {}".format(array1.flatten(), array2.flatten()))
        return False
