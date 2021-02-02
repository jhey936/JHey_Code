# coding=utf-8
"""Provides mutations based on rotations of parts of clusters"""
import numpy as np

from bmpga.mutation.basemutation import BaseMutation

from bmpga.storage import Cluster

from bmpga.utils.geometry import random_axis


class RandomSingleRotation(BaseMutation):
    """
    Applies a random rotation to one molecule in the cluster.
    """
    def __init__(self, initial_step_size: float=2 * np.pi) -> None:
        self.step_size = initial_step_size
        super().__init__(initial_step_size=initial_step_size, distribution="uniform")

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Applies a rotation to a single member of a cluster

        Args:
            cluster (Cluster): required, the cluster to be mutated
        """
        successful = False

        while not successful:
            selection = np.random.randint(len(cluster.molecules))

            # Step length is distributed according to the distribution selected at __init__
            theta = self.random(self.step_size)

            cluster.molecules[selection].rotate(axis=random_axis(), theta=theta)
            successful = self.check_cluster(cluster)
        return cluster


class RandomRotations(BaseMutation):
    """
    Applies random rotations to a random number of randomly selected initial_molecules in the cluster
    """
    def __init__(self, initial_step_size: float=2 * np.pi) -> None:
        self.step_size = initial_step_size
        super().__init__(initial_step_size=initial_step_size)

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Applies rotations to several members of a cluster

        Args:
            cluster (Cluster): required, the cluster to be mutated
        """
        successful = False

        while not successful:
            selection = np.random.randint(len(cluster.molecules), size=np.random.randint(len(cluster.molecules)))

            for particle in selection:

                theta = self.random(self.step_size)

                cluster.molecules[particle].rotate(axis=random_axis(), theta=theta)

            successful = self.check_cluster(cluster)

        return cluster



class Rock(BaseMutation):
    """
    Applies small random rotations to a all initial_molecules in the cluster
    """
    def __init__(self, initial_step_size: float=np.pi / 4.0):
        """

        Parameters
        ----------
        initial_step_size: maximum rotation. (Default = pi/4.0)
        """
        self.step_size = initial_step_size
        super().__init__(initial_step_size=initial_step_size, distribution="uniform")

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Applies small random rotations to a all initial_molecules in the cluster"""
        successful = False

        while not successful:

            for particle in range(len(cluster.molecules)):
                theta = self.random(self.step_size)
                cluster.molecules[particle].rotate(axis=random_axis(), theta=theta)

            successful = self.check_cluster(cluster)
        return cluster
