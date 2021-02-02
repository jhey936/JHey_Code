# coding=utf-8
"""Provides unittests for the C implementation of the LJ potential"""
import bmpga
import unittest

import numpy as np

from bmpga.potentials import LJcPotential, GeneralizedLennardJonesPotential
from bmpga.storage import Cluster, Molecule


# noinspection SpellCheckingInspection
class TestFastLJ(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up the fast LJ potential and the standard implementation as a reference"""
        self.ref_potential = GeneralizedLennardJonesPotential()
        test_data_dir = str(bmpga.__path__[0]) + "/tests/test_data/"
        self.lj38_coordinates = np.loadtxt(test_data_dir + "LJ38.xyz")

    def test_ljc_get_energy_minimum(self) -> None:
        fast_potential = LJcPotential(2)

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[2**(1./6.0), 0.0, 0.0]]),
                                                   particle_names=["LJ"])])

        ref_energy = self.ref_potential.get_energy(c1)
        test_energy = fast_potential.get_energy(c1)
        self.assertEqual(ref_energy, test_energy)

    def test_ljc_get_energy_0(self) -> None:
        fast_potential = LJcPotential(2)

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"])])

        # First we check that we get U=0 at r=1 and both implementations give the same result
        e_fast_0 = fast_potential.get_energy(c1)
        self.assertEqual(0.0, e_fast_0)
        self.assertEqual(self.ref_potential.get_energy(c1, ), e_fast_0)

    def test_ljc_get_energy_LJ38(self) -> None:
        lj38 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([c]),
                                                     particle_names=["LJ"])
                                            for c in self.lj38_coordinates])

        fast_potential = LJcPotential(38)
        # Check that both potentials calculate the same correct energy for the LJ38 minimum
        e_fast_38 = fast_potential.get_energy(lj38)
        self.assertAlmostEqual(-173.928427, e_fast_38, places=6)  # 6dp accuracy as this is reported online
        self.assertEqual(self.ref_potential.get_energy(lj38), e_fast_38)

    def test_ljc_get_jacobian_1(self) -> None:
        fast_potential = LJcPotential(2)

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[2**(1/6.), 0.0, 0.0]]),
                                                   particle_names=["LJ"])])

        test_jac = fast_potential.get_jacobian(c1.get_particle_positions()[0].flatten())

        # print(ref_jac, test_jac)
        self.assertEqual(np.array([-12.0, 0.0, 0.0, 12.0, 0.0, 0.0]).all(), test_jac.all())

    def test_ljc_jacobian(self) -> None:
        fast_potential = LJcPotential(38)

        lj38 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([c]),
                                                     particle_names=["LJ"])
                                            for c in self.lj38_coordinates])

        lj38_minimum = fast_potential.minimize(lj38)

        coords = lj38_minimum.get_particle_positions()

        coords = coords[0]

        lj38_min_jac = fast_potential.get_jacobian(coords.flatten())

        try:
            np.testing.assert_allclose(np.zeros(shape=38 * 3), lj38_min_jac, atol=0.001)

        except AssertionError:
            print(lj38_min_jac)
            print("This is expected to be close to 0 as it is close to a minimum")
            self.fail("Calculated jacobian not close to expected")

        # Check that both classes return the same jacobian
        ref_jac = self.ref_potential.get_jacobian(coordinates=coords.flatten())

        try:
            np.testing.assert_allclose(ref_jac, lj38_min_jac, atol=0.001)

        except AssertionError:
            print(lj38_min_jac, ref_jac)
            print("C implementation and pure python should return the same")
            self.fail("Calculated jacobian not close to expected")

    def test_ljc_minimizer(self) -> None:
        """Test that the minimizer returns the expected result"""
        fast_potential = LJcPotential(2)

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"])])

        min_ljc_small = fast_potential.minimize(cluster=c1)
        pos = min_ljc_small.get_particle_positions()
        coords = pos[0]
        min_dist_2_part = np.linalg.norm(coords[0]-coords[1])
        self.assertAlmostEqual((2 ** (1. / 6.)), min_dist_2_part)

    # def test_ljc_large_cluster(self) -> None:
    #     # check that we get the same result when optimizing a slightly perturbed structure
    #     perturbed_lj38_coordinates = self.lj38_coordinates + np.random.normal(scale=0.07, size=(38, 3))
    #
    #     min_ljc_38 = self.fast_potential.minimize(coordinates=copy.deepcopy(perturbed_lj38_coordinates), molecules=)
    #     min_lj_ref_38 = self.ref_potential.minimize(coordinates=copy.deepcopy(perturbed_lj38_coordinates))
    #     self.assertEqual(min_ljc_38['energy'], min_lj_ref_38['energy'])
    #
    # def test_ljc_jacobian_speed(self) -> None:
    #     """Tests that the c implementation is faster than the pure python one"""
    #
    #     coordinates = np.reshape(self.lj38_coordinates, newshape=self.lj38_coordinates.size)
    #     repeats = 3
    #
    #     def check_speed_ljc_jac() -> None:
    #         """Wraps without arguments to allow testing for speed"""
    #         self.fast_potential.get_jacobian(coordinates, )
    #
    #     def check_speed_lj_ref_jac() -> None:
    #         """Wraps without arguments to allow testing for speed"""
    #         self.ref_potential.get_jacobian(coordinates)
    #
    #     c_time = timeit.timeit(check_speed_ljc_jac, number=repeats)
    #     ref_time = timeit.timeit(check_speed_lj_ref_jac, number=repeats)
    #     # print("c_time = {}, ref_time = {}".format(c_time, ref_time))
    #     self.assertTrue(c_time < ref_time)
    #
    # def test_ljc_minimize_speed(self) -> None:
    #     """Tests that the c implementation is faster than the pure python one"""
    #
    #     coordinates = np.reshape(self.lj38_coordinates, newshape=self.lj38_coordinates.size)
    #     repeats = 3
    #
    #     def check_speed_ljc_minimize() -> None:
    #         """Wraps without arguments to allow testing for speed"""
    #         self.fast_potential.minimize(coordinates)
    #
    #     def check_speed_lj_ref_minimize() -> None:
    #         """Wraps without arguments to allow testing for speed"""
    #         self.ref_potential.minimize(coordinates)
    #
    #     c_time = timeit.timeit(check_speed_ljc_minimize, number=repeats)
    #     ref_time = timeit.timeit(check_speed_lj_ref_minimize, number=repeats)
    #
    #     self.assertTrue(c_time < ref_time)
