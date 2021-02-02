# coding=utf-8
"""Provide unittests for the storage classes such as the database and cluster objects"""
import unittest
import logging
import copy
import os

import numpy as np
import networkx as nx

from typing import Tuple

from bmpga.errors import CantIdentifyHeadTail

from bmpga.utils.io_utils import XYZReader
from bmpga.storage.database import Database, Cluster
from bmpga.storage.molecule import Molecule, FlexibleMolecule, IdentifyFragments
from bmpga.utils.testing_utils import check_list_almost_equal
from bmpga.characterization.simple_characterizer import SimpleEnergeticCharacterizer


class TestDatabase(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        """
        Sets up a database for testing
        """
        cls.db_location = "test_data/test.db"
        cls.delete_test_database(cls.db_location)
        cls.database = Database(db=cls.db_location,
                                new_database=True,
                                compare_clusters=SimpleEnergeticCharacterizer(accuracy=5e-7))
        cls.log = logging.getLogger(__name__)
        cls.log.info("DB setup complete")

    def test_database_creation_and_loading(self) -> None:
        self.assertTrue(self.database.is_bmpga_database)
        self.assertTrue(os.path.isfile(self.db_location))

        cluster1 = self.database.add_new_cluster(cost=5.0,
                                                 molecules=[Molecule(np.zeros(3))])

        c1_id = cluster1.id()
        # check that we can load database from disk
        database2 = Database(db=self.db_location)
        self.assertTrue(database2.is_bmpga_database)

        answer = database2.get_clusters_by_id(c1_id)

        self.log.debug(answer)

        self.assertIsInstance(answer, Cluster)
        self.assertEqual(answer.cost, 5.0)

    def test_add_cluster(self) -> None:
        out = self.database.add_new_cluster(cost=-10, molecules=[Molecule(np.zeros(3)), Molecule(np.ones(3))])
        self.assertIsInstance(out, Cluster)

    def test_insert_cluster(self) -> None:
        test_cluster = Cluster(cost=-44.44, molecules=[Molecule(np.zeros(3))])
        self.database.insert_cluster(test_cluster)

    def test_insert_clusters(self) -> None:
        c1 = Cluster(cost=-44.44, molecules=[Molecule(np.zeros(3))])
        c2 = Cluster(cost=-44.14, molecules=[Molecule(np.zeros(3))])

        clusters = [c1, c2]

        no_comparison_database = Database(new_database=True)

        self.log.debug(f"{no_comparison_database}, {type(no_comparison_database)}")

        self.assertIsInstance(no_comparison_database.fast_insert_clusters(clusters), type(None))

        self.assertEqual(no_comparison_database.minima[0].cost, -44.44)
        self.assertEqual(no_comparison_database.minima[1].cost, -44.14)

        # self.skipTest("Testing of database cluster insertion is not implemented")

    def test_no_comparison(self) -> None:
        c1 = Cluster(cost=-44.94, molecules=[Molecule(np.zeros(3))])

        no_comparison_database = Database(new_database=True)

        self.assertIsInstance(no_comparison_database.add_new_cluster(cost=-44.44, molecules=[Molecule(np.zeros(3))]),
                              Cluster)

        self.assertIsInstance(no_comparison_database.insert_cluster(c1), Cluster)

    def test_global_minimum(self) -> None:

        self.database.add_new_cluster(cost=-100.5,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])
        self.database.add_new_cluster(cost=-100.1,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])

        # Add clusters very close to the GM to test that cluster comparison is being called
        self.database.add_new_cluster(cost=-100.1+1e-7,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])
        self.database.add_new_cluster(cost=-100.1-1e-7,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])
        self.database.add_new_cluster(cost=-100.1-7e-7,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])

        global_minimum = self.database.get_global_minimum()
        self.assertEqual(global_minimum.cost, -100.5)

    def test_get_clusters_by_cost(self) -> None:
        self.database.add_new_cluster(cost=-100.5,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])
        self.database.add_new_cluster(cost=-11.5,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])
        self.database.add_new_cluster(cost=-12.5,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.zeros(3))])
        self.database.add_new_cluster(cost=-13.1,
                                      molecules=[Molecule(np.zeros(3)),
                                                 Molecule(np.ones(3))])

        clusters1 = self.database.get_clusters_by_cost(cost_max=-13.0)
        self.assertEqual(-100.5, clusters1[0].cost)
        self.assertEqual(-13.1, clusters1[-1].cost)

        clusters2 = self.database.get_clusters_by_cost(cost_min=-12.6, cost_max=-11.4)

        self.assertEqual(-12.5, clusters2[0].cost)
        self.assertEqual(-11.5, clusters2[1].cost)

        # Test that cost_min > cost_max raises an error
        with self.assertRaises(ValueError):
            self.database.get_clusters_by_cost(cost_max=-1.0, cost_min=2.0)

    def test_get_clusters_by_id(self) -> None:
        c1 = self.database.add_new_cluster(cost=-3.5,
                                           molecules=[Molecule(np.zeros(3)),
                                                      Molecule(np.ones(3))])
        c2 = self.database.add_new_cluster(cost=-3.6,
                                           molecules=[Molecule(np.zeros(3)),
                                                      Molecule(np.ones(3))])
        _id = c1.id()

        _ids = [_id, c2.id()]

        ret1 = self.database.get_clusters_by_id(_id)

        self.assertIsInstance(ret1, Cluster)
        self.assertEqual(ret1.cost, -3.5)

        ret2 = self.database.get_clusters_by_id(_ids)
        self.assertIsInstance(ret2, list)
        self.assertEqual(ret2[1].cost, -3.6)

    def test_cluster_id_equality_and_hash(self) -> None:
        c1 = self.database.add_new_cluster(cost=-1.5,
                                           molecules=[Molecule(np.zeros(3)),
                                                      Molecule(np.ones(3))])
        c2 = self.database.add_new_cluster(cost=-1.6,
                                           molecules=[Molecule(np.zeros(3)),
                                                      Molecule(np.ones(3))])

        # check id
        self.assertIsInstance(c1.id(), int)

        # Check equality vs instances of Cluster and integers
        c1_id = c1.id()
        self.assertFalse(c1 == c2)
        self.assertTrue(c1 == c1)
        self.assertTrue(c1 == c1_id)
        self.assertFalse(c2 == c1_id)

        # Check hashing function
        self.assertIsInstance(c1.__hash__(), int)

    def tearDown(self) -> None:
        """Removes the test database after testing"""
        self.delete_test_database(self.db_location)

    @staticmethod
    def delete_test_database(db_location) -> None:
        """Removes old database file if it exists"""
        if os.path.isfile(db_location):
            os.remove(db_location)


class TestClusterMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up some cluster objects for testing"""
        cls.cluster1 = Cluster(cost=-1.0, molecules=[Molecule(np.array([[-1., -1., -1.], [1., 1., 1.]]), ["H", "H"])])
        cls.cluster2 = Cluster(cost=-1.0, molecules=[Molecule(np.array([[-1., -1., -1.], [1., 1., 1.]]), ["H", "H"])])
        cls.cluster3 = Cluster(cost=-1.0, molecules=[Molecule(np.array([[-1., -1., -1.], [1., 1., 1.]]), ["H", "H"])])
        cls.log = logging.getLogger(__name__)

    def test_Cluster_get_molecular_positions(self) -> None:
        # get_molecular_positions returns the com of the initial_molecules
        self.assertListEqual(list(self.cluster1.get_molecular_positions()[0][0]), [0, 0, 0])

    def test_Cluster_get_particle_positions(self) -> None:
        # get_particle_positions returns the coordinates of the atoms
        self.assertListEqual(self.cluster1.get_particle_positions()[0][0].tolist(), [-1, -1, -1])

    def test_update_particle_positions(self) -> None:
        original_positions = self.cluster1.get_particle_positions()
        new_positions = list(copy.deepcopy(original_positions))
        self.log.debug(new_positions)
        new_positions[0] = np.array([[1., 1., 1.], [1., 1., 1.]])

        # noinspection PyTypeChecker
        self.cluster1.set_particle_positions(tuple(new_positions))
        self.log.debug(self.cluster1.get_particle_positions()[0].tolist())
        self.assertListEqual(self.cluster1.get_particle_positions()[0].tolist(), [[1.0, 1.0, 1.0]])

        self.cluster1.set_particle_positions(original_positions)
        self.assertListEqual(self.cluster1.get_particle_positions()[0].tolist(), [[-1, -1, -1]])

    def test_Cluster_translations(self) -> None:
        self.cluster2.translate(np.array([1.0, 1.0, 1.0]))
        self.assertListEqual(list(self.cluster2.get_molecular_positions()[0][0]), [1, 1, 1])

        self.cluster2.center()
        self.assertListEqual(list(self.cluster2.get_molecular_positions()[0][0]), [0, 0, 0])

    def test_cluster_rotation(self) -> None:
        self.cluster3.rotate(np.array([1., 1., 0.]), np.pi)
        self.assertTrue(check_list_almost_equal(self.cluster3.get_particle_positions()[0][0], [-1, -1, 1]))

        self.cluster3.rotate(np.array([0., 0., 1.]), np.pi)
        self.assertTrue(check_list_almost_equal(self.cluster3.get_particle_positions()[0][0], [1, 1, 1]))

        self.cluster3.rotate(np.array([1., 0., 0.]), np.pi/4.0)
        self.assertTrue(check_list_almost_equal(self.cluster3.get_particle_positions()[0][0], [1, np.sqrt(2), 0]))


class TestFlexibleMolecule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Sets up logging"""
        cls.log = logging.getLogger(cls.__name__)

    def setUp(self):
        """Sets up molcules for testing"""
        self.flex = FlexibleMolecule(coordinates=np.array([[0., 0., -0.3],
                                                           [0., 0., 1.], [0., 0., 2.],
                                                           [0., 0., 3.], [0., 1., 3.],
                                                           [0., 2., 3.], [0., 3., 3.],
                                                           [0., 3., 4.4], [0., 3., 1.8],
                                                           [0., 4., 3.], [0., 5.3, 3.],
                                                           [0., 0., 4.], [0., 0., 5.],
                                                           [0., 0., 6.], [0., 0., 7.3]]),

                                     particle_names=["LJ", "LJ", "LJ",
                                                     "LJ", "LJ", "LJ",
                                                     "LJ", "LJ", "LJ",
                                                     "LJ", "Be", "LJ",
                                                     "LJ", "LJ", "He"])

    def test_flexiblemolecule_init(self):
        flex = FlexibleMolecule(coordinates=np.array([[0., 0., -0.78022338],
                                                      [-0.77106958, 0., 0.29368804],
                                                      [0.77106958, 0., 0.29368804]]),
                                particle_names=["C", "O", "O"])
        self.assertIsInstance(flex, FlexibleMolecule)

    def test_get_head_tail(self):

        head_tail = self.flex.identify_head_tail()

        self.assertIsInstance(head_tail, Tuple)

    def test_get_head_tail_known_bad(self):
        """Calls test_get_head_tail with 3 possible heads"""

        with self.assertRaises(CantIdentifyHeadTail):
            # Molecule has all three atoms equidistant so can't pick two ends for the chain.
            FlexibleMolecule(coordinates=np.zeros((3, 3)), particle_names=["LJ"] * 3)

    def test_identify_chain(self):
        # ends = self.flex.identify_head_tail()
        chain = self.flex.identify_chain()

        self.assertEqual(len(chain), 7)

    def test_get_dihedrals(self):
        reader = XYZReader()

        cluster = reader.read("test_data/C5_zwitterion_W14.xyz")[0]
        molecule = cluster.molecules[0]
        self.log.debug(str(molecule)+"\n")

        molecule = FlexibleMolecule(coordinates=molecule.coordinates, particle_names=molecule.particle_names)

        np.testing.assert_almost_equal([-57.9721257, -61.21463831, -178.80030196,
                                        179.7081364, 63.18836517, 169.60954293],
                                       molecule.calculate_dihedrals(), decimal=5)


class TestMolecule(unittest.TestCase):

    def test_to_graph(self):

        self.skipTest("test_to_graph not yet implemented.")


class TestIdentifyFragments(unittest.TestCase):

    def test_graph_to_fragments__good_graph(self):

        mol_list = [Molecule(coordinates=np.zeros(3, 3), particle_names=["H", "H", "H"]),
                    Molecule(coordinates=np.ones(3, 3)*12, particle_names=["O", "O", "O"]),
                    ]
        IDfrags = IdentifyFragments(molecule_list=mol_list)

        mol_G = nx.Graph()

        mol_G.add_nodes_from([])
        print(IDfrags.graph_to_fragments())


if __name__ == "__main__":
    unittest.main()
