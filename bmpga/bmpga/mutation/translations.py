# coding=utf-8
"""Provides translation based mutations"""
import logging

import numpy as np

from bmpga.mutation import BaseMutation
from bmpga.storage import Cluster
from bmpga.utils.geometry import normalise


class RandomSingleTranslation(BaseMutation):
    """
    Applies a random translation to a single randomly selected molecule in the cluster
    """

    def __init__(self, initial_step_size: float = 2.0, distribution: str = "normal",
                 log: logging.Logger = None) -> None:
        """

        Args:
            initial_step_size: 
            distribution: 
            log: 
        """  # TODO: Document RandomSingleTranslation
        super().__init__(initial_step_size=initial_step_size, distribution=distribution, log=log)

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """

        Parameters
        ----------
        cluster: cluster object containing list of initial_molecules

        Returns
        -------

        """
        successful = False

        while not successful:
            # selection = np.random.randint(len(cluster.molecules))

            # Generates a uniformly random unit vector
            direction = normalise(np.random.uniform(-1, 1, size=3))

            # Step length is distributed according to the distribution selected at __init__
            step_length = self.random(self.step_size)

            step = direction * step_length
            # np.random.choice(cluster.molecules[selection])[0].translate(step)

            np.random.choice(cluster.molecules).translate(step)

            successful = self.check_cluster(cluster)
        return cluster


class RandomMultipleTranslations(BaseMutation):
    """
    Applies a random translation to a single randomly selected molecule in the cluster
    """

    def __init__(self, initial_step_size: float = 2.0, distribution: str = "normal",
                 log: logging.Logger = None) -> None:
        """

        Args:
            initial_step_size: 
            distribution: 
            log: 
        """  # TODO: Document RandomSingleTranslation
        super().__init__(initial_step_size=initial_step_size, distribution=distribution, log=log)

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """

        Parameters
        ----------
        cluster: cluster object containing list of initial_molecules

        Returns
        -------

        """
        successful = False

        while not successful:
            selection = np.random.randint(len(cluster.molecules), size=np.random.randint(len(cluster.molecules)))

            for particle in selection:
                # Generates a uniformly random unit vector
                direction = normalise(np.random.uniform(-1, 1, size=3))

                # Step length is distributed according to the distribution selected at __init__
                step_length = self.random(self.step_size)

                step = direction * step_length

                cluster.molecules[particle].translate(step)

            successful = self.check_cluster(cluster)

        return cluster


class Shake(BaseMutation):
    """Applies small random translations to all members of a cluster"""
    def __init__(self, initial_step_size: float=0.2, distribution="normal", log=None) -> None:
        self.step_size = initial_step_size
        super().__init__(initial_step_size=initial_step_size, distribution=distribution, log=log)

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Applies small random translations to all members of a cluster"""
        successful = False

        while not successful:

            for m in cluster.molecules:
                direction = normalise(np.random.uniform(-1, 1, size=3))
                step_length = self.random(self.step_size)
                step = direction * step_length
                m.translate(step)

            successful = self.check_cluster(cluster)
        return cluster
