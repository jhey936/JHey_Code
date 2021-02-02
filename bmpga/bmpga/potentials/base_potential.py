# coding=utf-8
"""
Base potential class.
New potentials should inherit from here.
See lennard_jones.py for an example of an implemented class
"""
from bmpga.storage import Cluster


class BasePotential(object):
    """The base object new potentials should inherit from.

    Methods:
        get_energy: Return the energy of the system at the current time
        minimize: minimize and return coordinates and energy of the system

    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_energy(self, cluster: Cluster, *args, **kwargs) -> float:
        """
        Method to return the single-point energy of a system

        Parameters
        ----------


        Returns
        -------
        float(energy): Energy of the system at the given coordinates

        Args:
            cluster: Cluster instance, required, the for which to calculate energy
            *args: list, optional, postitional arguments
            **kwargs: Other keyword arguments needed on a per-implementation basis (i.e. atom labels)

        """

        raise NotImplementedError

    def minimize(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Method to return the locally minimised energy of a system

        Attributes:
            coordinates: numpy.array(coordinates), N*3 array of atomic/particle coordinates
            **kwargs: Other keyword arguments needed on a per-implementation basis (i.e. atom labels)

        Returns:
            Cluster: minimised Cluster object
        """
        raise NotImplementedError
