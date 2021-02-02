# coding=utf-8
"""
Provides tests for the mating classes
"""
import copy
import logging
import unittest

import numpy as np

from bmpga.utils.geometry import magnitude
from bmpga.storage import Cluster, Molecule
from bmpga.mutation import RandomCluster, RandomSingleTranslation, Shake
from bmpga.utils.testing_utils import set_numpy_seed, check_list_almost_equal


class TestRandomCluster(unittest.TestCase):
    """Tests the RandomCluster mutation operator"""

    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls) -> None:
        """Sets up an instance of random cluster and a logger"""
        cls.log = logging.getLogger(__name__)
        cls.mutate = RandomCluster(log=cls.log)  # , box_length=10.0)

        cls.c1 = Cluster(cost=0.0,
                         molecules=[Molecule(coordinates=np.array([[1., 1, 1], [0, 0, 0]]),
                                             particle_names=["He", "H"]),
                                    Molecule(coordinates=np.array([[1., -1, -1], [0, 0, 0], [-1, -1, -1]]),
                                             particle_names=["H", "H", "He"])])

        cls.c2 = Cluster(cost=9.0,
                         molecules=[Molecule(coordinates=np.ones(shape=(2, 3)),
                                             particle_names=["B", "Be"]),
                                    Molecule(coordinates=np.zeros(shape=(3, 3)),
                                             particle_names=["Be", "B", "Be"])])

    def test_random_cluster(self) -> None:
        set_numpy_seed()

        new_cluster = self.mutate(self.c1)

        self.assertIsInstance(new_cluster, Cluster)

        # Check that we are not getting the same cluster back
        self.assertNotEqual(new_cluster, self.c1)

        # Check that we are getting the same masses back
        self.assertListEqual(new_cluster.molecules[0].masses.tolist(), [1.01, 1.01, 4.0])

        atomic_positions = new_cluster.get_particle_positions()[0]

        self.assertFalse(check_list_almost_equal(atomic_positions, self.c1.get_particle_positions()[0]))

    # def test_random_cluster_fixed_box(self) -> None:
    #     set_numpy_seed()
    #
    #     random_cluster = RandomCluster(box_length=10, fixed_box=True, log=self.log)
    #     random_cluster.mutate(self.c1)
    #     self.log.debug(parse_info_log())
    #     self.assertTrue(
    #         b"Returning random cluster found after 1 attempts using a final box_length of 10."
    #         in parse_info_log())


class TestRandomSingleTranslation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up logging"""
        cls.log = logging.getLogger(__name__)

    def setUp(self) -> None:
        """Sets the numpy seed to a known value"""
        set_numpy_seed()

    def test_RST_init_default(self) -> None:
        mutation = RandomSingleTranslation()
        self.assertIsInstance(mutation, RandomSingleTranslation)
        self.assertEqual(mutation.step_size, 2.0)

    def test_RST_init_step_size(self) -> None:
        mutation = RandomSingleTranslation(initial_step_size=4.0)
        self.assertIsInstance(mutation, RandomSingleTranslation)
        self.assertEqual(mutation.step_size, 4.0)

    def test_RST_init_good_distribution(self) -> None:
        mutation = RandomSingleTranslation(distribution="uniform")
        self.assertIsInstance(mutation, RandomSingleTranslation)
        self.assertEqual(mutation.step_size, 2.0)

    def test_RST_init_bad_distribution(self) -> None:
        with self.assertRaises(NotImplementedError):
            RandomSingleTranslation(distribution="not a real distribution")

    def test_RST_mutate_good(self) -> None:

        c1 = Cluster(cost=9.0,
                     molecules=[Molecule(coordinates=np.random.uniform(low=10, size=(2, 3)),
                                         particle_names=["B", "Be"]),
                                Molecule(coordinates=np.random.uniform(low=10, size=(3, 3)),
                                         particle_names=["Be", "B", "Be"])])

        mutation = RandomSingleTranslation()
        new_cluster = mutation.mutate(copy.deepcopy(c1))

        self.assertIsInstance(new_cluster, Cluster)
        self.assertFalse(np.allclose(new_cluster.get_molecular_positions()[0], c1.get_molecular_positions()[0]))


class TestShake(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up logging"""
        cls.log = logging.getLogger(__name__)

    def setUp(self) -> None:
        """Sets the numpy seed to a known value"""
        set_numpy_seed()

    def test_RST_init_default(self) -> None:
        mutation = Shake()
        self.assertIsInstance(mutation, Shake)
        self.assertEqual(mutation.step_size, 0.2)

    def test_RST_init_step_size(self) -> None:
        mutation = Shake(initial_step_size=0.4)
        self.assertIsInstance(mutation, Shake)
        self.assertEqual(mutation.step_size, 0.4)

    def test_RST_init_good_distribution(self) -> None:
        mutation = Shake(distribution="uniform")
        self.assertIsInstance(mutation, Shake)
        self.assertEqual(mutation.step_size, 0.2)

        # self.skipTest("Need method to test uniform distribution")


    def test_shake_mutate_good(self) -> None:
        c1 = Cluster(cost=9.0,
                     molecules=[Molecule(coordinates=np.random.uniform(low=10, size=(2, 3)),
                                         particle_names=["B", "Be"]),
                                Molecule(coordinates=np.random.uniform(low=10, size=(3, 3)),
                                         particle_names=["Be", "B", "Be"])])

        mutation = Shake()

        mutated_c1 = mutation.mutate(copy.deepcopy(c1))

        diff = c1.get_particle_positions()[0] - mutated_c1.get_particle_positions()[0]

        self.assertEqual(magnitude(diff[0]), 0.42059300827254525)
        self.assertEqual(magnitude(diff[-1]), 0.4186786088973787)
