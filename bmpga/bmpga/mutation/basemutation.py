# coding=utf-8
"""Provides the base mutation object"""
import copy
import logging

import numpy as np

from bmpga.storage import Cluster


class BaseMutation(object):
    """
    Base class defining the structure of a mutation class.
    New mutations should inherit from this class.
    """

    def __init__(self, initial_step_size: float,
                 distribution: str = "normal",
                 cutoff: float = 0.6,
                 log: logging.Logger = None) -> None:
        """Sets up the log, step size and the distribution

        Note: normal distribution invokes numpy.random.normal(step_size, step_size/2) so gives a
                distribution

        Args:
            initial_step_size: float, required, the initial step size to use
            distribution: str, optional, the distribution to use. Currently only "normal" and "uniform" are implemented
                             (default="normal")
            log: logging.Logger instance, optional, your project logger (default=logging.Logger(__name__))
        """

        self.log = log or logging.getLogger(__name__)

        self.step_size = initial_step_size
        self.cutoff = cutoff**2

        if distribution == "normal":
            self.random = lambda x: np.random.normal(x, x/2.0)
        elif distribution == "uniform":
            self.random = lambda x: np.random.uniform(0, x)
        else:
            message = "{} Not implemented!".format(distribution)
            try:
                raise NotImplementedError(message)
            except NotImplementedError as error:
                self.log.exception(error)
                raise

    def update(self, factor) -> None:
        """Method to update the step size to adjust acceptance rate"""
        self.step_size *= factor

    def __call__(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        self.log.debug(f"Mutating using: {self.__class__.__name__}")
        return copy.deepcopy(self.mutate(cluster, *args, **kwargs))

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Method called every time a mutation is requested"""
        raise NotImplementedError

    def check_cluster(self, cluster: Cluster) -> bool:
        """Simple distance check to ensure that there is no particle overlap"""

        all_r_sq = (cluster.get_particle_positions()[0]**2).sum(-1)

        if min(all_r_sq) <= self.cutoff:
            return False
        else:
            return True

