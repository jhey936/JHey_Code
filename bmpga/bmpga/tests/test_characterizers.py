# coding=utf-8
"""Provides unittests for characterizers"""
import unittest

import numpy as np

from bmpga.storage import Cluster, Molecule
from bmpga.characterization.simple_characterizer import SimpleEnergeticCharacterizer, SimpleGeometricCharacterizer


class TestCharacterizers(unittest.TestCase):

    def test_simple_energy_characterizer(self) -> None:
        """Tests the simplest characterizer, based simply on energy"""
        c1 = Cluster(cost=-100, molecules=[])
        c2 = Cluster(cost=-99.9, molecules=[])
        c3 = Cluster(cost=-99.9+1e-6, molecules=[])
        c4 = Cluster(cost=-99.9+2e-6, molecules=[])

        # noinspection PyArgumentEqualDefault
        simple_characterizer = SimpleEnergeticCharacterizer(accuracy=1e-6)  # default accuracy=1e-6 shown for clarity

        # c1 == c1
        self.assertTrue(simple_characterizer(c1, c1))
        # c2 is very close in energy to c3
        self.assertTrue(simple_characterizer(c2, c3))
        # c1 and c2 are distant in energy
        self.assertFalse(simple_characterizer(c1, c2))
        # c2 and c4 are above the threshold apart from each other
        self.assertFalse(simple_characterizer(c2, c4))

    def test_simple_geometric_characterizer_same_cluster(self) -> None:

        compare = SimpleGeometricCharacterizer()

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 1.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 0.0, -1.0]]), particle_names=["LJ"])])

        c2 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[-1.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, -1.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 0.0, 1.0]]), particle_names=["LJ"])])

        self.assertTrue(compare(c1, c2))

    def test_simple_geometric_characterizer_v_diff_cluster(self) -> None:

        compare = SimpleGeometricCharacterizer()

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 1.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 0.0, -1.0]]), particle_names=["LJ"])])

        c2 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[-1.0, 0.3, 7.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[2.0, -1.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.3, 1.0, 10.0]]), particle_names=["LJ"])])

        self.assertFalse(compare(c1, c2))

    def test_simple_geometric_characterizer_nearly_same_cluster_fail(self) -> None:

        compare = SimpleGeometricCharacterizer()

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 1.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 0.0, -1.0]]), particle_names=["LJ"])])

        c2 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[-1.0, 0.1, -0.1]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, -1.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.1, -0.1, 1.1]]), particle_names=["LJ"])])

        self.assertFalse(compare(c1, c2))

    def test_simple_geometric_characterizer_similar_cluster_succeed(self) -> None:

        compare = SimpleGeometricCharacterizer()

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 1.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.0, 0.0, -1.0]]), particle_names=["LJ"])])

        c2 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[-1.0, 2e-1, 1e-1]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[5e-2, -1.0, 1e-1]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[1e-2, 0.0, 1.1]]), particle_names=["LJ"])])

        self.assertTrue(compare(c1, c2))
