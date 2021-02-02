# coding=utf-8
"""Provides unittests for the potentials"""
import os
import bmpga
import shutil
import logging
import unittest

import numpy as np

from bmpga.utils.geometry import magnitude
from bmpga.storage import Cluster, Molecule

from bmpga.potentials.OPLS_ff import OPLS_potential
from bmpga.potentials.dft.nwchem_potential import NWChemPotential
from bmpga.potentials.dft.base_dft_potential import BaseDFTPotential
from bmpga.potentials.LJ_potential import GeneralizedLennardJonesPotential


class TestLJPot(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets up two different Lennard-Jones potentials for testing.
        A 6-12 potential and a 15-30 potential.
        """
        # noinspection PyArgumentEqualDefault
        cls.LJ_6_12 = GeneralizedLennardJonesPotential(base_exponent=6)
        cls.LJ_15_30 = GeneralizedLennardJonesPotential(base_exponent=20)

    def test_lj_energy_r1(self) -> None:

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]), particle_names=["LJ"])])
        # First we check that we get V=0 at r=1
        self.assertEqual(0.0, self.LJ_6_12.get_energy(c1))
        self.assertEqual(0.0, self.LJ_15_30.get_energy(c1))

    def test_lj_energy_r_min(self) -> None:
        # Then check we get v~-1 @ r~r_{min} (r_min = 2**(1./exp_attractive))
        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[2**(1./6.0), 0.0, 0.0]]),
                                                   particle_names=["LJ"])])

        c2 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[2 ** (1. / 20.), 0.0, 0.0]]),
                                                   particle_names=["LJ"])])

        self.assertAlmostEqual(-1.0, self.LJ_6_12.get_energy(c1))
        self.assertAlmostEqual(-1.0, self.LJ_15_30.get_energy(c2))

    def test_minimise_dimer(self) -> None:
        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[2**(1./6.0), 0.0, 0.0]]),
                                                   particle_names=["LJ"])])

        min1 = self.LJ_6_12.minimize(c1)

        self.assertTrue(min1['success'])
        self.assertAlmostEqual((2**(1./6.)), magnitude(min1['coordinates'][0], min1['coordinates'][1]))

    def test_minimize_3LJ(self) -> None:

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]),
                                                   particle_names=["LJ"]),
                                          Molecule(coordinates=np.array([[0.5, 0.5, 0.0]]),
                                                   particle_names=["LJ"])])

        min2 = self.LJ_6_12.minimize(c1)

        self.assertAlmostEqual(-3, min2['energy'])

    def test_minimize_LJ38(self) -> None:

        lj38_coordinates = np.loadtxt(bmpga.__path__[0]+"/tests/test_data/LJ38.xyz")

        c1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([coord]),
                                                   particle_names=["LJ"]) for coord in lj38_coordinates])

        min_lj38 = self.LJ_6_12.minimize(c1)

        self.assertAlmostEqual(-173.928427, min_lj38['energy'], places=6)


class TestBaseDFTPotential(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up logging for TestBaseDFTPotential"""
        cls.log = logging.getLogger(__name__)
        cls.test_path = bmpga.__path__[0]+"/tests/test_data/"
        cls.template_file = cls.test_path+"ENERGY_test_NWChem_template.nw"
        cls.output_file = cls.test_path+"template_test_out.nw"

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        """Removes output files"""
        if os.path.exists(cls.output_file):
            os.remove(cls.output_file)

    def test_insert_to_template(self) -> None:

        potential = BaseDFTPotential(run_string="nonsense", work_dir=os.curdir)

        coordinates = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
        labels = ["He", "He", "He"]

        target_dict = {"<NAME>": "TEST", "<XYZ>": potential.format_XYZ(coordinates, labels)}

        potential.insert_to_template(template=self.template_file,
                                     out_file=self.output_file,
                                     target_dict=target_dict)

        result = False
        with open(self.output_file) as out_f:
            for line in out_f.readlines():
                if "He" in line and "-1.0" in line:
                    result = True
                    break

        self.assertTrue(result)


class TestNWChemPotential(unittest.TestCase):
    """Tests the implementation of thee NWChem dft potential"""
    @classmethod
    def setUpClass(cls) -> None:
        """Sets up logging, and some data paths for the test, as well as a base potential for testing"""
        cls.log = logging.getLogger(__name__)
        cls.test_data_path = bmpga.__path__[0]+"/tests/test_data/"
        minimize_template = cls.test_data_path+"MINIMIZE_test_NWChem_template.nw"
        # noinspection PyPep8Naming
        SPE_template = cls.test_data_path+"ENERGY_test_NWChem_template.nw"
        cls.nwchem_potential = NWChemPotential(minimize_template=minimize_template,
                                               energy_template=SPE_template,
                                               run_string="mpiexec -np 2 nwchem",
                                               work_dir=cls.test_data_path,
                                               )

    def test_NWChem_potential_initialise(self) -> None:
        self.assertTrue(self.nwchem_potential)

    def test_NWChem_potential_minimize(self) -> None:
        c1: Cluster = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[-1.1, 1.1, 0.2]]),
                                                            particle_names=["He"]),
                                                   Molecule(coordinates=np.array([[1.0, 0.0, 0.1]]),
                                                            particle_names=["He"]),
                                                   Molecule(coordinates=np.array([[0.1, -2.0, 0.0]]),
                                                            particle_names=["He"])])

        if shutil.which("nwchem") is None:
            self.skipTest("No NWChem binary found on system")
        else:
            returned_cluster = self.nwchem_potential.minimize(c1)

            returned_coords = returned_cluster.get_particle_positions()
            self.assertListEqual(returned_coords[2], ["He", "He", "He"])

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        """Removes any directories made during testing"""
        for idx in range(1, int(1e6)):
            path = f"{cls.test_data_path}NWCHEM{idx}"
            if os.path.exists(path):
                cls.log.debug(f"Removing {path}")
                shutil.rmtree(path)
            else:
                break


class TestOPLS(unittest.TestCase):

    def test_init_good(self):
        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1])
        self.assertIsInstance(pot, OPLS_potential)

        # noinspection PyArgumentEqualDefault
        pot2 = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1], interaction_mask=None)
        self.assertIsInstance(pot2, OPLS_potential)

    # noinspection PyUnusedLocal
    def test_init_bad_attr_array_lens(self):
        with self.assertRaises(AttributeError):
            pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1])

    # noinspection PyUnusedLocal
    def test_init_bad_interaction_mask_dtypes(self):
        with self.assertRaises(AttributeError):
            pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1], interaction_mask=np.array(["a", "b"]))
        with self.assertRaises(AttributeError):
            pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1], interaction_mask=np.array([["a"], ["b"]]))
        with self.assertRaises(AttributeError):
            pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1],
                                 interaction_mask=np.array([1.0, 0.0], dtype=np.float32))
        with self.assertRaises(AttributeError):
            pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1],
                                 interaction_mask=np.array([1, 0], dtype=np.int8))

    def test_init_good_interaction_mask(self):

        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1],
                             interaction_mask=np.array([1], dtype=np.int64))
        self.assertIsInstance(pot, OPLS_potential)

        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1],
                             interaction_mask=np.array([1.0], dtype=np.float64))
        self.assertIsInstance(pot, OPLS_potential)

        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1],
                             interaction_mask=np.array([True], dtype=np.bool))
        self.assertIsInstance(pot, OPLS_potential)

    # noinspection PyUnusedLocal
    def test_init_bad_interaction_mask_len(self):
        with self.assertRaises(AttributeError):
            pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1],
                                 interaction_mask=np.array([True, True], dtype=np.bool))
        with self.assertRaises(AttributeError):
            pot = OPLS_potential(q=[0, 0, 0], eps=[1, 1, 1], sigma=[1, 1, 1],
                                 interaction_mask=np.array([True, True], dtype=np.bool))

    def test_get_energy_GLJ_zero(self):
        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1])

        # GLJ energy at r_ij=1.0 == 0.0 eV
        clus1 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                                                      particle_names=["LJ", "LJ"])])

        self.assertEqual(pot.get_energy(clus1), 0.0)

        # At large r, en ~~ 0.0
        clus2 = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e99]]),
                                                      particle_names=["LJ", "LJ"])])

        self.assertEqual(pot.get_energy(clus2), 0)

    def test_get_energy_GLJ_minimum(self):
        # GLJ energy at r_ij~1.112 ~~ -1.0 eV

        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1])

        clus = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2**(1/6)]]),
                                                     particle_names=["LJ", "LJ"])])
        self.assertEqual(pot.get_energy(clus), -1)

    def test_get_energy_GLJ_close_contact(self):
        # At small r, en >> 0.0

        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1])

        clus = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
                                                     particle_names=["LJ", "LJ"])])

        self.assertGreater(pot.get_energy(clus), 1e3)

    def test_get_energy_coulombic_zero_charges(self):

        pot = OPLS_potential(q=[0, 0], eps=[0, 0], sigma=[1, 1])

        clus = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
                                                     particle_names=["LJ", "LJ"])])

        self.assertEqual(pot.get_energy(clus), 0)

    def test_get_energy_coulombic_like_charges(self):

        pot = OPLS_potential(q=[1, 1], eps=[0, 0], sigma=[1, 1])

        clus = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                                                     particle_names=["LJ", "LJ"])])

        self.assertGreater(pot.get_energy(clus), 0)

    def test_get_energy_coulombic_unlike_charges(self):

        pot = OPLS_potential(q=[1, -1], eps=[0, 0], sigma=[1, 1])

        clus = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
                                                     particle_names=["LJ", "LJ"])])

        self.assertLess(pot.get_energy(clus), 0)

    def test_minimize_GLJ_only(self):
        pot = OPLS_potential(q=[0., 0.0], eps=[1., 1.], sigma=[1., 1.])

        clus = Cluster(cost=0.0,
                       molecules=[Molecule(coordinates=np.array([[0.1, 0.1, 0.], [0.2, 0.2, 1.122]]),
                                           particle_names=["LJ", "LJ"])])

        res = pot.minimize(clus)

        self.assertTrue(res['success'])
        self.assertEqual(res["energy"], -1.0)

    def test_minimize_full(self):
        pot = OPLS_potential(q=[0.5, -0.5], eps=[1., 1.], sigma=[1., 1.])

        clus = Cluster(cost=0.0,
                       molecules=[Molecule(coordinates=np.array([[0.1, 0.1, 0.], [0.2, 0.2, 1.122]]),
                                           particle_names=["LJ", "LJ"])])

        res = pot.minimize(clus)
        print(res)
        self.assertTrue(res['success'])
        self.assertLess(res["energy"], -1.0)

    def test_jacobian_0(self):
        pot = OPLS_potential(q=[0, 0], eps=[1, 1], sigma=[1, 1])

        jacobian = pot.get_jacobian(np.array([[0., 0., 0.], [0., 0., 2 ** (1. / 6.)]]).flatten())
        self.assertLess(abs(min(jacobian)), 1e-10)

    # #todo Add tests for the interaction mask

    def test_tip4p_dimer(self):

        clus = Cluster(molecules=[Molecule()])
