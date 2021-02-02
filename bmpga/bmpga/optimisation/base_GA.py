# coding=utf-8
"""
Module containing the base pool class
"""


class BaseGA(object):
    """
    The base class that other Pool classes should inherit from.

    Attributes:
        database: bmpga Database, required, the initialised Database object containing

    Required methods:
        cull_pool
        mating
        write_pool_to_file
        initialise_pool

    """
    def __init__(self, *args, **kwargs) -> None:
        """Initialises the GA at instantiation time

        Returns:
            None

        """
        pass

    def cull_pool(self) -> list:
        """Method should return a list of cluster objects that represents the full population at the time of calling

        Returns:
            clusters: list of Cluster objects
        """
        raise NotImplementedError

    def select_clusters(self, n_parents: int) -> list:
        """Method to return selected parents from the pool

        Returns:
            parents, tuple of Cluster representing the parents selected for mating
        """
        raise NotImplementedError

    def write_pool_to_file(self, filename: str = None) -> None:
        """Writes the current pool to a file"""
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        """Method called every update_freq number of steps"""
        pass
