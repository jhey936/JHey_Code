# coding=utf-8
"""Tests for stochastic gradient descent"""

import unittest

import numpy as np

from bmpga.storage import Cluster, Molecule
from bmpga.potentials import LJcPotential

from bmpga.utils.io_utils import XYZWriter
from bmpga.optimisation.stochastic_gradient import SGD

class NewLJ(LJcPotential):
    """Potential for testing"""

    def __init__(self, n):
        super().__init__(n)

    def _calc_g(self, rsq):

        ir2 = 1./rsq
        ir6 = ir2**3
        ir12 = ir6 ** 2

        g = -4.0 * ir2 * ((12.0 * ir12) - (6.0 * ir6))
        return g


class TestSGD(unittest.TestCase):

    def setUp(self):
        """Sets the seed to a known value"""
        np.random.seed(1)

    def test_setup(self):
        self.assertTrue(SGD(potential=NewLJ(3)))

    def test_SGD_runs(self):

        writer = XYZWriter()
        pot = NewLJ(5)
        c1 = Cluster(molecules=[Molecule(coordinates=np.array([[0.566474720473, 0.05189774298450, 0.03347914367068]]),
                                         particle_names=["Cl"]),
                                Molecule(coordinates=np.array([[-0.010999390189086, 1.01397142828, -1.00418537828]]),
                                         particle_names=["Cl"]),
                                Molecule(coordinates=np.array([[-0.555475330284, 0.0341308287337, 0.0623354819510]]),
                                         particle_names=["Cl"]),
                                Molecule(coordinates=np.array([[-0.39523644062, 2.8659668824697, 0.3990951103299]]),
                                         particle_names=["Cl"]),
                                Molecule(coordinates=np.array([[-0.39523644062, 0.8659668824697, 0.3990951103299]]),
                                         particle_names=["Cl"])
                                ],)
        sgd = SGD(potential=pot)

        c_ret = sgd(c1)
        writer.write(c_ret, "test_data/out.xyz")
        self.assertIsInstance(c_ret, Cluster)
        self.assertAlmostEqual(pot.get_energy(c_ret), -9.10385, places=3)


    def test_test_pot(self):
        pot = NewLJ(13)
        self.assertEqual(pot._calc_g(1), -24)
        self.assertTrue(pot._calc_g(1.2599210499) < 0.00000001)
