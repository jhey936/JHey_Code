# coding=utf-8
"""Provides tests for classes in bmpga.systems"""
import copy
import bmpga
import unittest

import numpy as np

from bmpga.systems import DefineSystem
from bmpga.storage import Cluster, IdentifyFragments, Molecule

from bmpga.utils.testing_utils import check_list_almost_equal


class TestDefineSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the test_data path"""
        cls.test_data = bmpga.__path__[0]+"/tests/test_data/"

    def test_DefineSystem_get_template_molecules_xyz_1structure(self) -> None:
        system = DefineSystem(numbers_of_molecules=[1], template_files=self.test_data + "water_template.xyz")
        self.assertEqual(len(system.initial_molecules), 1)
        self.assertListEqual(system.initial_molecules[0].particle_names, ["O", "H", "H"])
        self.assertEqual(system.initial_molecules[0].name, "HHO")

    def test_DefineSystem_get_template_molecules_xyz_1structure_3repeats(self) -> None:
        system = DefineSystem(numbers_of_molecules=[3], template_files=self.test_data + "water_template.xyz")
        self.assertEqual(len(system.initial_molecules), 1)
        self.assertListEqual(system.initial_molecules[0].particle_names, ["O", "H", "H"])
        self.assertEqual(system.initial_molecules[0].name, "HHO")
        cluster = system.get_random_cluster()
        print(f"\n{cluster}{cluster.molecules}\n")
        print(cluster.get_molecular_positions())


    def test_DefineSystem_get_template_molecules_xyz_3structs(self) -> None:
        system = DefineSystem(numbers_of_molecules=[1, 1, 1], template_files=self.test_data + "water_template_3.xyz")
        self.assertEqual(len(system.initial_molecules), 3)
        self.assertListEqual(system.initial_molecules[1].particle_names, ["H", "H", "O"])
        self.assertEqual(system.initial_molecules[2].name, "HHO")

    def test_DefineSystem_get_template_molecules_multiple_xyz(self) -> None:

        file_names = [self.test_data + "water_template.xyz", self.test_data + "water_template_3.xyz"]
        system = DefineSystem(numbers_of_molecules=[1, 1, 1, 1], template_files=file_names)
        self.assertEqual(len(system.initial_molecules), 4)
        self.assertListEqual(system.initial_molecules[2].particle_names, ["H", "H", "O"])
        self.assertEqual(system.initial_molecules[3].name, "HHO")

    def test_DefineSystem_get_template_molecules_pdb(self) -> None:

        with self.assertRaises(NotImplementedError):
            DefineSystem([1], template_files=self.test_data + "water_template_3.pdb")

    def test_DefineSystem_get_template_molecules_mol2(self) -> None:

        with self.assertRaises(NotImplementedError):
            DefineSystem([1], template_files=self.test_data + "water_template_3.mol2")

    def test_DefineSystem_init_with_molecules(self) -> None:

        mol1 = Molecule(coordinates=np.zeros(shape=(2, 3)), particle_names=["D", "H"])
        mol2 = Molecule(coordinates=np.ones(shape=(2, 3)), particle_names=["He", "He"])

        system = DefineSystem([1, 1], molecules=[mol1, mol2])
        self.assertListEqual(list(system.initial_molecules[0].coordinates[0]), [0, 0, 0])
        self.assertEqual(system.initial_molecules[1].name, "HeHe")

    def test_DefineSystem_no_molecules_no_template(self) -> None:

        with self.assertRaises(AssertionError):
            DefineSystem([1])

    def test_DefineSystem_get_random_cluster(self) -> None:
        mol1 = Molecule(coordinates=np.array([[1., 1., 1.], [0., 0., 0.]]), particle_names=["D", "H"])
        mol2 = Molecule(coordinates=np.array([[1., 1., 1.], [0., 0., 0.]]), particle_names=["He", "He"])

        system = DefineSystem([1, 2], molecules=[mol1, mol2])
        self.assertListEqual(list(system.initial_molecules[1].coordinates[1]), [0, 0, 0])
        self.assertEqual(system.initial_molecules[1].name, "HeHe")

        self.assertIsInstance(system.get_random_cluster(), Cluster)

    def test_call_DefineSystem_returns_a_random_cluster(self) -> None:
        mol1 = Molecule(coordinates=np.array([[1., 1., 1.], [0., 0., 0.]]), particle_names=["D", "H"])
        mol2 = Molecule(coordinates=np.array([[1., 1., 1.], [0., 0., 0.]]), particle_names=["He", "He"])

        system = DefineSystem([1, 2], molecules=[mol1, mol2])
        self.assertListEqual(list(system.initial_molecules[0].coordinates[1]), [0, 0, 0])
        self.assertEqual(system.initial_molecules[1].name, "HeHe")

        self.assertIsInstance(system(), Cluster)

    def test_DefineSystem_unequal_molecules_and_numbers(self) -> None:
        mol1 = Molecule(coordinates=np.array([[1., 1., 1.], [0., 0., 0.]]), particle_names=["D", "H"])
        mol2 = Molecule(coordinates=np.array([[1., 1., 1.], [0., 0., 0.]]), particle_names=["He", "He"])

        with self.assertRaises(AssertionError):
            DefineSystem([2], molecules=[mol1, mol2])


class TestMolecule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up some test initial_molecules"""
        cls.mol1 = Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), particle_names=["C", "H"])
        cls.mol2 = Molecule(coordinates=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), particle_names=["H", "H"])
        cls.mol3 = Molecule(coordinates=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
        cls.mol4 = Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), particle_names=["C", "H"])
        cls.mol5 = Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]]), particle_names=["H", "H"])
        cls.mol6 = Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]),
                            particle_names=["H", "H", "H"])
        cls.mol7 = Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]),
                            masses=[1.6, 1.3, 1.2])
        cls.mol8 = Molecule(coordinates=np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, -1.0]]),
                            particle_names=["H", "C"])

    def test_names_simple(self) -> None:
        # If labels are given, name is set to string of sorted labels
        self.assertEqual(self.mol1.name, "CH")
        self.assertEqual(self.mol2.name, "HH")
        self.assertEqual(self.mol4.name, "CH")
        self.assertEqual(self.mol5.name, "HH")
        self.assertEqual(self.mol6.name, "HHH")

    def test_name_sorting_strings(self) -> None:
        # Names should be sorted ["H", "C"] -> "CH"
        self.assertEqual(self.mol8.name, "CH")

    def test_names_only_masses(self) -> None:
        # If only masses are given, name is set to sorted masses
        self.assertEqual(self.mol7.name, "1.21.31.6")

    def test_no_particle_labels_or_masses(self) -> None:
        # If neither masses or labels is given name is set to "Unknown"
        self.assertEqual(self.mol3.name, "Unknown")

    def test_assignment(self) -> None:
        """Test that atomic masses are assigned properly using self.mol1"""
        self.assertEqual([12.01, 1.01], list(self.mol1.masses))
        self.assertEqual(["C", "H"], list(self.mol1.particle_names))
        self.assertTrue(np.allclose([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], list(self.mol1.coordinates)))

    def test_validate(self) -> None:
        """Test that the Molecule validates inputs"""

        self.assertTrue(Molecule(np.zeros(shape=(3, 3))))
        with self.assertRaises(AttributeError):
            Molecule(np.zeros(shape=(3, 3)), particle_names=["H"])

    def test_com_000(self) -> None:
        """Test that the center of mass is properly calculated for molecule5"""
        com = self.mol2.get_center_of_mass()
        self.assertListEqual([0.0, 0.0, 0.0], list(com))

    def test_com_not_origin(self) -> None:
        com = self.mol5.get_center_of_mass()
        self.assertListEqual([0.5, -0.5, 0.5], list(com))

    def test_COM_no_atom_labels(self) -> None:
        """
        Tests that errors are raised as expected.

        """
        self.assertListEqual(list(self.mol3.get_center_of_mass()), [0, 0, 0])


    def test_center(self) -> None:
        """
        Tests that centering is performed correctly

        """
        self.mol6.center()
        self.assertListEqual([0.0, 0.0, 0.0], list(self.mol6.get_center_of_mass()))

    def test_translate(self) -> None:
        """
        Tests that translations are applied properly

        """

        self.mol4.translate(np.array([0.0, 0.0, 1.0]))
        self.assertListEqual([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
                             [list(c) for c in self.mol4.coordinates])
        # Translate back
        self.mol4.translate(np.array([0.0, 0.0, 0.0]))

    def test_mol_eq_self_self(self) -> None:
        """Tests that the hashing function works"""
        self.assertEqual(self.mol1, self.mol1)  # self == self

    def test_mol_eq_self_copy_of_self(self) -> None:
        mol1_1 = copy.deepcopy(self.mol1)
        self.assertEqual(self.mol1, mol1_1)  # check that identical copies evaluate to equal

    def test_mol_eq_not_self(self) -> None:
        self.assertNotEqual(self.mol1, self.mol2)  # self != not self

    def test_mol_eq_not_type_mol(self) -> None:
        self.assertNotEqual(self.mol2, 3)  # self != another type

    def test_molecular_rotations(self) -> None:
        """
        Tests for correct applications of molecular

        """
        mol = Molecule(np.array([np.zeros(3), np.ones(3)]), ["H", "H"])

        # Rotate by pi about the xy axis
        mol.rotate(np.array([1, 1, 0]), np.pi)
        self.assertTrue(check_list_almost_equal(mol.coordinates, [[0, 0, 1], [1, 1, 0]]))

        # Rotate back to original positions
        mol.rotate(np.array([1, 1, 0]), np.pi)
        self.assertTrue(check_list_almost_equal(mol.coordinates, [[0, 0, 0], [1, 1, 1]]))

        # Rotate about x
        mol.rotate(np.array([1, 0, 0]), np.pi)
        self.assertTrue(check_list_almost_equal(mol.coordinates, [[0, 1, 1], [1, 0, 0]]))

        # Rotate about xz
        mol.rotate(np.array([1, 0, 1]), np.pi)
        self.assertTrue(check_list_almost_equal(mol.coordinates, [[1, 0, 0], [0, 1, 1]]))


class TestIdentifyFragments(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up an identifyFragments instance"""
        cls.identifier = IdentifyFragments()  # default error = 1.05

    def test_init_identify_fragments_basic(self) -> None:
        self.assertTrue(IdentifyFragments())  # Tests that the default __init__ works

    def test_init_identify_fragments_setting_error(self) -> None:
        # Tests that assignment is working
        self.assertEqual(IdentifyFragments(error=5.0).error, 5.0)

    def test_init_identify_fragments_setting_non_int_error(self) -> None:
        # Tests that passing anything but a float to error breaks it
        with self.assertRaises(AssertionError):
            IdentifyFragments(error=1)

    def test_init_identify_fragments_setting_radii(self) -> None:
        # Check that not passing an instance of element_utils.ParticleRadii throws an error
        from bmpga.utils.elements import ParticleRadii
        rad = ParticleRadii()
        IdentifyFragments(radii=rad)

    def test_init_identify_fragments_setting_radii_not_instance_ParticleRadii(self) -> None:
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            IdentifyFragments(radii=4)

    def test_get_cutoff_distance_identify_fragments(self) -> None:

        # Test the base case
        r_cut = self.identifier.get_cutoff_distance("H", "C")
        self.assertAlmostEqual(r_cut, 1.14*1.05)

    def test_cutoff_distance_caching(self) -> None:
        self.identifier.get_cutoff_distance("H", "C")

        # Test to see if result was cached
        self.assertAlmostEqual(self.identifier.cutoffs["HC"], 1.14 * 1.05)  # default error = 1.05
        self.assertAlmostEqual(self.identifier.cutoffs["CH"], 1.14 * 1.05)
