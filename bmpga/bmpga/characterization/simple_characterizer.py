# coding=utf-8
"""Provides a really simple characterizer"""
import numpy as np

from bmpga.storage import Cluster
from bmpga.characterization.base_characterizer import BaseCharacterizer
from bmpga.characterization.characterisation_utils import align_clusters, gyration_tensor


class SimpleEnergeticCharacterizer(BaseCharacterizer):
    """Provides really simple comparisons of clusters based solely on energy"""
    def __init__(self, accuracy: float = 1e-7, log=None) -> None:

        self.accuracy = accuracy
        super().__init__(log)
        self.log.warning("""Using SimpleEnergeticCharacterizer is not recommended. 
        Uniqueness is only based on energy so energetically degenerate structures will be discarded.""")

    def compare(self, cluster1: Cluster, cluster2: Cluster, *args, **kwargs):
        """Compares clusters based solely on their cost functions. This is poor. Don't use it.
        
        Args:
            cluster1: Cluster, required
            cluster2: Cluster, required
            *args: other positional arguments
            **kwargs: other keyword arguments

        Returns:
            True if clusters are within self.accuracy energy of one another.
            False if clusters are sufficiently different

        """
        delta_energy = abs(cluster1.cost - cluster2.cost)
        return delta_energy <= self.accuracy


class SimpleGeometricCharacterizer(BaseCharacterizer):
    """
    Characterizes clusters based soley on geometry
    """
    def __init__(self, accuracy: float = 1e-1):
        """

        Args:
            accuracy, (float): optional, the accuray below which the clusters are considered the same. (default=1e-1)
        """
        self.accuracy = accuracy
        super().__init__()

    def compare(self, cluster1: Cluster, cluster2: Cluster, *args, **kwargs) -> bool:
        """Simple geometry comparison.

        Aligns clusters, then

        Args:
            cluster1, (Cluster): required
            cluster2, (Cluster): required

        Returns:
            bool, True if the clusters are the same to within self.accuracy, else False

        """
        align_clusters(cluster1, cluster2)
        g_tensor_c1 = gyration_tensor(cluster1.get_particle_positions()[0])
        g_tensor_c2 = gyration_tensor(cluster2.get_particle_positions()[0])
        g_tensor_c1 -= g_tensor_c2

        return np.linalg.norm(g_tensor_c1) <= self.accuracy

class EnergyGeometryCharacterizer(BaseCharacterizer):
    """
    Characterizes clusters based on both geometry and energy
    """
    def __init__(self, energy_accuracy: float = 1e-7, gyration_accuracy: float = 1e-1) -> None:
        self.energy = SimpleEnergeticCharacterizer(energy_accuracy)
        self.gyration = SimpleGeometricCharacterizer(gyration_accuracy)
        super().__init__()

    def compare(self, cluster1: Cluster, cluster2: Cluster, *args, **kwargs) -> bool:
        """
        Returns true if the energy and the gyration tensor BOTH fall within their respective tolerences.
        (True if the clusters are the same)
        """
        return self.energy(cluster1, cluster2) and self.gyration(cluster1, cluster2)

