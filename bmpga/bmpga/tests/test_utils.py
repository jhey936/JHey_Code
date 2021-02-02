# coding=utf-8
"""Provides unittests for some classes/methods in the bmpga.utils module"""
import os
import bmpga
import logging
import unittest

import numpy as np

from bmpga.storage import Cluster
from bmpga.utils.io_utils import XYZReader, BaseWriter, XYZWriter
from bmpga.storage.molecule import Molecule
from bmpga.utils.testing_utils import parse_info_log
from bmpga.utils.elements import ParticleRadii
from bmpga.utils.chem import get_masses, get_mass
from bmpga.utils.geometry import get_rotation_matrix, get_all_magnitudes, get_dihedral, find_center_of_mass, get_angle


class TestChemUtils(unittest.TestCase):
    """Class to test functions in utils.chem_utils"""
    def test_get_masses_H2(self) -> None:
        self.assertListEqual([1.01, 1.01], list(get_masses(["H", "H"])))

    def test_get_masses_benzene(self) -> None:
        self.assertListEqual([12.01, 12.01, 12.01, 12.01, 12.01, 12.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01],
                             list(get_masses(["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"])))

    def test_KeyError(self) -> None:
        with self.assertRaises(KeyError):
            get_masses(["InvalidKey"])

    def test_TypeError(self) -> None:
        """Should raise a TypeError when no atom labels are supplied"""
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            get_masses()

    def test_get_mass(self) -> None:
        mol1 = Molecule(np.array([[0, 0, 0], [1, 1, 1]]), ["H", "H"])
        mol2 = Molecule(np.array([[0, 0, 0], [1, 1, 1]]), ["C", "H"])

        self.assertEqual(2.02, get_mass(mol1))
        self.assertEqual(13.02, get_mass(mol2))

        mol2.masses = None
        self.assertEqual(13.02, get_mass(mol2))


class TestGeomUtils(unittest.TestCase):

    class DummyCluster:
        """Dummy object to test find_center_of_mass"""
        def __init__(self, coordinates: np.ndarray, masses: list=None, particle_names: list=None) -> None:
            self.coordinates = coordinates
            self.masses = masses
            self.particle_names = particle_names

    def test_get_dihedral_0(self) -> None:
        dihedral_0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertEqual(get_dihedral(dihedral_0), 0)

    def test_get_dihedral_180(self) -> None:
        # Check that we can get 180
        dihedral_180 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 1E-14]])
        self.assertAlmostEqual(get_dihedral(dihedral_180), np.pi)

    def test_get_dihedral_90(self) -> None:
        # Check 90
        dihedral_90 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
        self.assertEqual(get_dihedral(dihedral_90), np.pi/2)

    def test_get_dihedral_45(self) -> None:
        # Check 45
        dihedral_45 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1, 0.5]])
        self.assertEqual(get_dihedral(dihedral_45), np.pi/4)

    def test_get_dihedral_signing(self) -> None:
        # Check that the signs are correct
        dihedral_sign = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1, -0.5]])
        self.assertEqual(get_dihedral(dihedral_sign), -np.pi/4)

    def test_get_dihedral_improper_shape(self) -> None:
        # Check that incorrect shapes raise the correct error
        with self.assertRaises(ValueError):
            get_dihedral(np.zeros(shape=(3, 3)))

    def test_get_dihedral_degrees(self) -> None:
        # Check we can get degrees if we want them
        dihedral_45 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1, 0.5]])
        self.assertEqual(get_dihedral(dihedral_45, radians=False), 45)

    def test_find_center_of_mass_basic(self) -> None:

        masses1 = [1, 1]
        coordinates1 = np.array([[1, 1, 1], [-1, -1, -1]])
        dc1 = self.DummyCluster(coordinates1, masses1)
        self.assertListEqual(list(find_center_of_mass(cluster=dc1)), [0, 0, 0])

    def test_find_center_of_mass_atom_names(self) -> None:
        coordinates1 = np.array([[1, 1, 1], [-1, -1, -1]])
        # Test handling of clusters
        dc_no_mass = self.DummyCluster(coordinates1, particle_names=['H', "H"])
        self.assertListEqual(list(find_center_of_mass(cluster=dc_no_mass)), [0, 0, 0])

    def test_find_center_of_mass_DUMMY_atom(self) -> None:
        # Test dummy atom
        coordinates1 = np.array([[1, 1, 1], [-1, -1, -1]])
        dc_dummy_atom = self.DummyCluster(coordinates1, masses=[1, 0])
        self.assertEqual(list(find_center_of_mass(cluster=dc_dummy_atom)), [1, 1, 1])

    def test_get_all_magnitudes(self) -> None:

        coordinate_array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        all_magnitudes = get_all_magnitudes(coordinate_array)

        self.assertAlmostEqual(all_magnitudes[0], np.sqrt(3))
        self.assertAlmostEqual(all_magnitudes[1], 2*np.sqrt(3))
        self.assertAlmostEqual(all_magnitudes[2], np.sqrt(3))

    def test_rot_mat(self) -> None:
        # Checks that a 0pi rotation returns correctly
        self.assertTrue(np.allclose([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    get_rotation_matrix(np.array([0, 0, 1]), 0)))
        # Checks that a 2pi rotation returns correctly
        self.assertTrue(np.allclose(get_rotation_matrix(np.array([0, 0, 1]), 0,),
                                    get_rotation_matrix(np.array([0, 0, 1]), 2 * np.pi)))
        # Checks that a 1pi rotation returns correctly
        self.assertTrue(np.allclose(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                                    get_rotation_matrix(np.array([1, 0, 0]), np.pi)))

    def test_get_angle_45(self) -> None:
        """Tests that get_angle returns the correct"""
        angle_45 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(get_angle(angle_45), np.pi/4)

    def test_get_angle_90(self) -> None:
        angle_90 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        self.assertEqual(get_angle(angle_90), np.pi/2)

    def test_get_angle_degrees(self) -> None:
        angle_90 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        self.assertEqual(get_angle(angle_90, radians=False), 90.0)

    def test_get_angle_180(self) -> None:
        angle_180 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.assertEqual(get_angle(angle_180), np.pi)

    def test_get_angle_0(self) -> None:
        angle_0 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(get_angle(angle_0), 0)

    def test_gyration_tensor(self) -> None:
        """Tests the gyration_tensor utility"""
        # TODO: test gyration tensor
        self.skipTest("test_gyration_tensor not yet implemented")


class TestParticleRadii(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the test data path"""
        cls.test_data_path = bmpga.__path__[0] + "/tests/test_data/"
        cls.radii = ParticleRadii()  # Suppress printing during tests

    def test_atomic_number_mapping(self) -> None:

        self.assertEqual(self.radii._atomic_number_dict[1], "H")
        self.assertEqual(self.radii._atomic_number_dict[4], "Be")
        self.assertEqual(self.radii._atomic_number_dict[79], "Au")
        self.assertEqual(self.radii._atomic_number_dict[200], "LJ")

    def test_initialise_particle_radii(self) -> None:
        # Setup a default instance of ParticleRadii
        self.assertTrue(ParticleRadii())

    def test_initialise_particle_radii_custom_csv(self) -> None:

        # initialise from a known-good custom csv
        radii = ParticleRadii(csv_path=self.test_data_path + "good_custom_atomic_radii.csv")
        self.assertTrue(radii)
        self.assertEqual(radii.particle_radii.loc["He", 'Covalent'], 2.0)

    def test_initialise_particle_radii_custom_csv_bad_paths(self) -> None:
        # Check that bad paths correctly raise errors
        with self.assertRaises(OSError):
            ParticleRadii(csv_path="bad_path_to_a.csv")

        with self.assertRaises(OSError):
            ParticleRadii(csv_path="very/bad_path_to_a.csv")

    def test_overrides_particle_radii(self) -> None:
        # Test overrides

        override_list = [["H", "Covalent", 1.0], ["Be", "Ionic", 0.22], [79, "Crystal", 12.0]]
        overridden_radii = ParticleRadii(overrides=override_list)

        self.assertEqual(overridden_radii.particle_radii.loc['H'].Covalent, 1.0)
        self.assertEqual(overridden_radii.particle_radii.loc['Be'].Ionic, 0.22)
        self.assertEqual(overridden_radii.particle_radii.loc['Au'].Crystal, 12.0)

    def test_get_radius_covalent(self) -> None:

        # Test that get_radius returns the correct radius for an element with a known covalent radius
        self.assertEqual(self.radii.get_radius("Ge"), 1.22)  # Ge Covalent_radius = 1.22A

    def test_get_radius_with_blank_values(self) -> None:
        # Tests that blank values in the csv are correctly skipped,
        self.assertEqual(self.radii.get_radius("Cm"), 1.11)  # 1.11 is the 'Crystal' radius for Cm

    def test_get_radius_by_atomic_number(self) -> None:
        # Tests that radii can be found using atomic number
        self.assertEqual(self.radii.get_radius(79), 1.44)  # Au Covalent_radius = 1.44A

    def test_get_radius_request_empty_value(self) -> None:
        # Tests that specific self.radii can be retrieved and ValueError is raised on returning an empty cell
        self.assertEqual(self.radii.get_radius("Po", "Atomic"), 1.35)  # Po Atomic_radius = 1.35A
        with self.assertRaises(ValueError):
            self.radii.get_radius("Po", "Covalent")  # Not in csv

    def test_get_radius_non_existent_label(self) -> None:
        # Test that nonsense labels return a KeyError
        with self.assertRaises(KeyError):
            self.radii.get_radius("SomeSillySymbol")

    def test_get_radius_bad_line_in_csv(self) -> None:
        # Check that bad lines raise errors properly
        radii = ParticleRadii(csv_path=self.test_data_path + "good_custom_atomic_radii.csv")
        with self.assertRaises(ValueError):
            radii.get_radius("Li")


class TestXYZReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the xyz reader and provides a path to some test data"""
        cls.reader = XYZReader()
        test_data_dir = str(bmpga.__path__[0]) + "/tests/test_data/"
        cls.water12_file = test_data_dir + "water12.xyz"
        cls.log = logging.getLogger(__name__)

    def test_read_XYZ(self) -> None:
        clusters = self.reader.read(self.water12_file)
        self.assertIsInstance(clusters[0], Cluster)
        self.assertIsInstance(clusters[1], Cluster)

    def test__call__xyz_reader(self) -> None:
        clusters = self.reader(self.water12_file)
        self.assertIsInstance(clusters[0], Cluster)
        self.assertIsInstance(clusters[1], Cluster)

    def test_xyz_reader_clean(self) -> None:
        data = np.array(["9876  123 \n", "        qwerty 1234  \n    "])
        clean_data = self.reader.clean(data)
        self.assertEqual(clean_data[0][0], "9876")
        self.assertEqual(clean_data[0][1], "123")
        self.assertEqual(clean_data[1][0], "qwerty")
        self.assertTrue(len(clean_data[1]) == 2)

    def test_xyz_reader_reshape(self) -> None:
        data = np.zeros(18)
        data = self.reader.reshape(data, 4)
        self.assertTrue(data.shape == (3, 6))

    def test_xyz_reader_xyz2coordinates(self) -> None:
        structures = self.reader(self.water12_file, return_clusters=False)
        # self.log.debug(structures[0][0][0][0])
        self.assertListEqual(list(structures[1][0][0][0]), [-1.5888764851, 1.56273974365, 2.71849443552])


class TestLogging(unittest.TestCase):

    def setUp(self) -> None:
        """Setup logger for testing"""
        self.log = logging.getLogger(__name__)

    def test_debug(self) -> None:

        self.log.debug("testing_debug")
        self.log.info("testing_info")
        self.log.warning("testing_warning")
        self.log.error("testing_error")

        # Test exception handling:
        try:
            raise ValueError
        except ValueError:
            self.log.exception("testing_exception")

        with open("info.log") as f:
            data = f.readlines()

        for log_type, line in zip(["DEBUG", "INFO", "WARNING", "ERROR", "ERROR", "Traceback"][::-1], data[-4::-1]):
            self.assertTrue(log_type in line)


class TestBaseWriter(unittest.TestCase):
    def test_parse_structures_single_cluster(self) -> None:
        c1 = Cluster(cost=0.0,
                     molecules=[Molecule(coordinates=np.zeros(shape=(2, 3)), particle_names=["H", "He"]),
                                Molecule(coordinates=np.zeros(shape=(3, 3)), particle_names=["H", "H", "He"])])

        output = self.writer._parse_structures(c1)
        self.assertListEqual(output[0][0][0].tolist(), np.zeros(shape=(5, 3)).tolist())  # Check coords are correct

    def test_parse_structures_list_clusters(self) -> None:
        c1 = Cluster(cost=0.0,
                     molecules=[Molecule(coordinates=np.zeros(shape=(2, 3)), particle_names=["H", "He"]),
                                Molecule(coordinates=np.zeros(shape=(3, 3)), particle_names=["H", "H", "He"])])

        c2 = Cluster(cost=9.0,
                     molecules=[Molecule(coordinates=np.ones(shape=(2, 3)), particle_names=["B", "Be"]),
                                Molecule(coordinates=np.ones(shape=(3, 3)), particle_names=["Be", "B", "Be"])])
        cluster_list = [c1, c2]

        output = self.writer._parse_structures(cluster_list)
        self.assertListEqual(output[0][1][0].tolist(), np.zeros(shape=(5, 3)).tolist())  # Check coords are correct
        self.assertListEqual(output[1], [0, 9])

    def test__get_coord_labels(self) -> None:

        c1 = Cluster(cost=0.0,
                     molecules=[Molecule(coordinates=np.zeros(shape=(2, 3)), particle_names=["H", "He"]),
                                Molecule(coordinates=np.zeros(shape=(3, 3)), particle_names=["H", "H", "He"])])

        c2 = Cluster(cost=0.0,
                     molecules=[Molecule(coordinates=np.ones(shape=(2, 3)), particle_names=["B", "Be"]),
                                Molecule(coordinates=np.ones(shape=(3, 3)), particle_names=["Be", "B", "Be"])])
        cluster_list = [c1, c2]
        coord_labels = self.writer._get_coord_labels(cluster_list)

        self.assertListEqual(list(coord_labels[0][0][0]), [0, 0, 0])
        self.assertTrue("Be" in coord_labels[1][1])

    def test__get_coord_labels_1_cluster(self) -> None:
        c1 = Cluster(cost=0.0,
                     molecules=[Molecule(coordinates=np.zeros(shape=(2, 3)), particle_names=["H", "He"]),
                                Molecule(coordinates=np.zeros(shape=(3, 3)), particle_names=["H", "H", "He"])])

        cluster_list = [c1]
        coord_labels = self.writer._get_coord_labels(cluster_list)

        self.assertListEqual(list(coord_labels[0][0][0]), [0., 0., 0.])
        self.assertTrue("He" in coord_labels[0][1])

    @classmethod
    def setUpClass(cls) -> None:
        """Define a logger"""
        cls.log = logging.getLogger(__name__)
        cls.writer = BaseWriter(log=cls.log)

    def test_check_file_name_good_exists_append(self) -> None:
        """Checks that _check_file_name is working correctly"""

        fn = "test.file"
        with open(fn, "w") as f:
            f.write("test_string")

        # Check that appending raises the correct warning
        self.assertTrue(self.writer._check_file_name(filename=fn, file_mode="a", n_structs=1) is None)
        self.log.debug("Last 150b of log = " + str(parse_info_log()) + "\n")
        warning = b"WARNING - File test.file already exists. 1 new structures will be appended to this file."
        self.assertTrue(warning in parse_info_log())

        if os.path.exists(fn):
            os.remove(fn)

    def test_check_file_name_good_exists_overwrite(self) -> None:

        fn = "test.file"
        with open(fn, "w") as f:
            f.write("test_string")

        # Check that overwriting raises the correct warning
        self.assertTrue(self.writer._check_file_name(filename=fn, file_mode="w", n_structs=1) is None)

        log_info = str(parse_info_log().decode())
        self.log.debug("Last 150b of log = " + log_info + "\n")
        warning = "WARNING - File test.file exists! This file has been moved to"
        self.assertTrue(warning in str(log_info))

        tmp_file = "".join([x for x in log_info.split() if "test.file" in x][1])
        os.remove(tmp_file)

        if os.path.exists(fn):
            os.remove(fn)

    def test_check_file_name_good_new_file(self) -> None:

        fn = "test.file"
        if os.path.exists(fn):
            os.remove(fn)
        # Check that writing to an empty file raises the correct info message in the log
        self.assertTrue(self.writer._check_file_name(filename=fn, file_mode="a", n_structs=1) is None)
        self.log.debug("Last 150b of log = " + str(parse_info_log()) + "\n")
        self.assertTrue(b"Creating new file test.file, and writing 1 structure(s)" in parse_info_log())


class TestXYZWriter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Define a logger"""
        cls.log = logging.getLogger(__name__)

    def test_XYZ_writer_write(self) -> None:
        test_fn = "test.xyz"
        coord_labels = [[np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), ["H", "H"]]]
        writer = XYZWriter()

        writer(coord_labels, filename=test_fn)

        self.assertTrue(os.path.exists(test_fn))

        with open(test_fn, "rb") as f:
            file_data = f.read()

        os.remove(test_fn)
        self.log.info(file_data)
        self.assertTrue(
            b'2\nStructure 0, Energy None, id None\nH    0.0    0.0    0.0\nH    1.0    1.0    1.0\n' == file_data
        )

    def test_XYZ_writer__format(self) -> None:
        un_formatted = [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), ["H", "H"]]
        writer = XYZWriter()

        formatted = writer._format(un_formatted)
        self.log.debug(formatted)
        self.assertListEqual(formatted, ['2\n', '\n', 'H    0.0    0.0    0.0\n', 'H    1.0    1.0    1.0\n'])
