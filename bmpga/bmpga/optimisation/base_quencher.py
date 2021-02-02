# coding=utf-8
"""The base class used to perform local optimisations"""
import os

from bmpga.storage import Cluster

from bmpga.potentials.base_potential import BasePotential


class BaseQuencher(object):
    """
    The base class for performing local optimisations.
    Other quenchers should inherit from this class

    """
    def __init__(self, potential: BasePotential, *args, **kwargs) -> None:
        self.potential = potential

    def minimize(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Method which performs local optimisation by making calls to the potential

        Args:
            cluster: Cluster instance, required, Cluster to be optimised. Usually the result of mating or mutation
            *args: optional, positional arguments to be passed to the potential
            **kwargs: optional, keyword arguments to be passed to the potential

        Returns:
            A the quenched Cluster object. (This will be the same cluster with updated cost and coordinates)

        """
        raise NotImplementedError

    def get_energy(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Method which performs single point energy evaluation by making calls to the potential

        Args:
            cluster: Cluster instance, required, Cluster to be optimised. Usually the result of mating or mutation
            *args: optional, positional arguments to be passed to the potential
            **kwargs: optional, keyword arguments to be passed to the potential

        Returns:
            A the updated Cluster object. (This will be the same cluster with updated cost and coordinates)

        """
        raise NotImplementedError

    def run(self) -> None:
        """
        Main method of class.
        Runs a fixed number of quenches and then terminates.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        # noinspection SpellCheckingInspection
        return f"""QuenchClient running on Host: {os.uname()[1]} With pid: {os.getpid()}"""
