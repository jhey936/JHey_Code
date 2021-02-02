# coding=utf-8
"""Provides unittests for the various methods of selecting parents"""
import unittest
import logging
import copy

from typing import List

import numpy as np

from bmpga.mating import RouletteWheelSelection, RankSelector, BoltzmannSelector, TournamentSelector, DeavenHoCrossover
from bmpga.storage import Cluster, Molecule

from bmpga.utils.testing_utils import set_numpy_seed, get_dummy_population, DummyPopMember


class TestRankSelector(unittest.TestCase):
    def setUp(self) -> None:
        """Sets the random seed and sets up the dummy population and Rank selector"""
        set_numpy_seed()
        self.selector = RankSelector()
        self.population = get_dummy_population()

    def test_rank_call(self) -> None:
        result = self.selector(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def test_rank_get_parents(self) -> None:
        result = self.selector.get_parents(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def test_rank_weighting(self) -> None:
        population = [DummyPopMember(-1000), DummyPopMember(-900), DummyPopMember(-800), DummyPopMember(-1)]
        results = [self.selector(population) for _ in range(1000)]

        a, b = 0, 0
        for parents in results:
            self.assertNotEqual(parents[0].cost, parents[1].cost)  # extra check to ensure parents are not chosen twice
            if parents[0].cost == -1000 or parents[1].cost == -1000:
                a += 1
            else:
                b += 1

        self.assertTrue(a > b)


class TestRouletteWheel(unittest.TestCase):
    def setUp(self) -> None:
        """
        Sets up the roulette wheel selector and a dummy population
        """
        set_numpy_seed()
        self.population = [DummyPopMember(f) for f in np.linspace(-10, 0, 10)]
        self.selector = RouletteWheelSelection()

    def test_roulette_call(self) -> None:
        result = self.selector(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def test_roulette_get_parents(self) -> None:
        result = self.selector.get_parents(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def test_roulette_weighting(self) -> None:
        population = [DummyPopMember(-1000), DummyPopMember(-900), DummyPopMember(-800), DummyPopMember(-1)]
        results = [self.selector(population) for _ in range(1000)]

        a, b = 0, 0
        for parents in results:
            self.assertNotEqual(parents[0].cost, parents[1].cost)  # extra check to ensure parents are not chosen twice
            if parents[0].cost == -1000 or parents[1].cost == -1000:
                a += 1
            else:
                b += 1
        self.assertTrue(a > b)


class TestBoltzmannSelector(unittest.TestCase):
    def setUp(self) -> None:
        """
        Sets up the boltzmann weighted selector and a dummy population
        """
        set_numpy_seed()
        self.population = [DummyPopMember(f) for f in np.linspace(-10.0, -9.0, 10)]
        self.selector = BoltzmannSelector(kb=0.0019872041, temperature=130)  # using kb in kcal/mol*K

    def test_boltzmann_call(self) -> None:
        result = self.selector(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def test_boltzmann_get_parents(self) -> None:
        result = self.selector.get_parents(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def test_boltzmann_distribution(self) -> None:
        # TODO: come up with a better test for distributions
        # self.skipTest("not_working")
        population = get_dummy_population(-100, -90, 1000)

        parents = []

        for i in range(500):
            p1, p2 = self.selector(population)
            parents.append(p1.cost)
            parents.append(p2.cost)

        parent_costs = np.array(parents)

        values, _ = np.histogram(parent_costs)
        self.assertListEqual([515, 250, 108, 59, 31, 17, 11, 5, 3, 1], list(values))


class TestTournamentSelector(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets up the tournament selector and a small dummy population
        """
        set_numpy_seed()
        cls.population = [DummyPopMember(f) for f in np.linspace(-10.0, -1.0, 10)]
        cls.selector = TournamentSelector()

    def test_tournament_call(self) -> None:
        set_numpy_seed()
        result = self.selector(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def test_tournament_get_parents(self) -> None:
        set_numpy_seed()
        result = self.selector.get_parents(self.population)
        self.assertIsInstance(result, List)
        self.assertIsInstance(result[0], DummyPopMember)

    def fetch_distribution(self, population) -> np.ndarray:
        """Gets a large array of parents to check the distribution is correct"""
        parents = []

        for i in range(500):
            p1, p2 = self.selector(population)
            parents.append(p1.cost)
            parents.append(p2.cost)

        parent_costs = np.array(parents)

        values, _ = np.histogram(parent_costs)
        return values

    def test_tournament_distribution(self) -> None:

        # Test default distribution
        set_numpy_seed()
        population = get_dummy_population(-100, -90, 1000)

        values_def = self.fetch_distribution(population)

        self.assertListEqual([547, 243, 110, 41, 34, 11, 9, 2, 0, 3], list(values_def))

    def test_tournament_distribution_stochastic(self) -> None:

        # Test stochastic distribution when k=1
        set_numpy_seed()
        population = get_dummy_population(-100, -90, 1000)

        self.selector.k = 1
        values_stochastic = self.fetch_distribution(population)
        self.selector.__init__()  # reinitialise to reset k to default
        self.assertTrue((20-np.sum(np.log10(values_stochastic))) <= 0.02)

    def test_tournament_distribution_deterministic(self) -> None:

        # Test deterministic tournament selection
        set_numpy_seed()
        population = get_dummy_population(-100, -90, 1000)

        values_def = self.fetch_distribution(population)
        self.selector.p = 1  # Set p_select for the fittest member in the tournament to 1
        values_deterministic = self.fetch_distribution(population)
        self.selector.__init__()  # reinitialise to reset p to default
        self.assertTrue(values_deterministic[0] > values_def[0])


class TestDHCrossover(unittest.TestCase):
    """Tests Devan-Ho single and multipoint mating"""
    @classmethod
    def setUpClass(cls) -> None:
        """Sets up Deaven Ho mating for testing"""

        cls.log = logging.getLogger(__name__)
        cls.mate = DeavenHoCrossover(max_attempts=100)

        cls.seed = 1

        cls.c1 = Cluster(cost=-3.5,
                         molecules=[Molecule(np.array([[0., 0, 0], [0, 4, 0]]), ["H", "H"]),
                                    Molecule(np.array([[0., 0, 1], [0, 3, 0]]), ["H", "H"]),
                                    Molecule(np.array([[0., 0, 2], [0, 2, 0]]), ["Cr", "H"]),
                                    Molecule(np.array([[0., 0, 3], [0, 1, 0]]), ["H", "H"]),
                                    Molecule(np.array([[0., 0, 4], [0, 0, 0]]), ["H", "H"])])

        cls.c2 = Cluster(cost=-3.5,
                         molecules=[Molecule(np.array([[4., 0, 0], [0, 0, 4]]), ["H", "H"]),
                                    Molecule(np.array([[3., 0, 1], [1, 0, 2]]), ["H", "H"]),
                                    Molecule(np.array([[2., 0, 2], [2, 0, 2]]), ["H", "Cr"]),
                                    Molecule(np.array([[1., 0, 3], [3, 0, 1]]), ["H", "H"]),
                                    Molecule(np.array([[0., 2, 4], [4, 0, 1]]), ["H", "H"])])

        cls.c3 = Cluster(cost=-3.5,
                         molecules=[Molecule(np.array([[0., 1, 0], [5, 4, 3]]), ["H", "H"]),
                                    Molecule(np.array([[1., 2, 1], [4, 3, 4]]), ["H", "H"]),
                                    Molecule(np.array([[2., 3, 2], [3, 2, 5]]), ["Cr", "H"]),
                                    Molecule(np.array([[3., 4, 3], [2, 1, 6]]), ["H", "H"]),
                                    Molecule(np.array([[4., 5, 4], [1, 0, 7]]), ["H", "H"])])

    def setUp(self):
        """Sets the seed to a known value before each test is run"""
        np.random.seed(self.seed)
        self.seed += 1

    def test_init(self) -> None:
        self.assertTrue(DeavenHoCrossover())

    def test_multi_point_crossover(self) -> None:
        c1 = copy.deepcopy(self.c1)
        c2 = copy.deepcopy(self.c3)

        child = self.mate([c1, c2], n_crossover_points=2)

        self.assertFalse(c1.get_molecular_positions()[1] == child.get_molecular_positions()[1])
        self.assertFalse(c2.get_molecular_positions()[1] == child.get_molecular_positions()[1])

    def test_single_point_crossover(self) -> None:
        c1 = copy.deepcopy(self.c1)
        c2 = copy.deepcopy(self.c2)
        set_numpy_seed()

        child = self.mate([c1, c2], n_crossover_points=1)

        self.assertFalse(c1.get_molecular_positions()[1] == child.get_molecular_positions()[1])
        self.assertFalse(c2.get_molecular_positions()[1] == child.get_molecular_positions()[1])

    def test_3_parent_crossover(self) -> None:
        c1 = copy.deepcopy(self.c1)
        c2 = copy.deepcopy(self.c2)
        c3 = copy.deepcopy(self.c3)
        set_numpy_seed()

        child = self.mate([c1, c2, c3], n_crossover_points=2)

        self.assertFalse(c1.get_molecular_positions()[1] == child.get_molecular_positions()[1])
        self.assertFalse(c2.get_molecular_positions()[1] == child.get_molecular_positions()[1])
        self.assertFalse(c3.get_molecular_positions()[1] == child.get_molecular_positions()[1])

    def test_error_checking_1_parent(self) -> None:
        c1 = copy.deepcopy(self.c1)
        set_numpy_seed()

        # Test assertionError is raised with too few parents
        with self.assertRaises(AssertionError):
            self.mate([c1], n_crossover_points=1)

    def test_error_checking_no_crossover(self) -> None:
        c1 = copy.deepcopy(self.c1)
        c2 = copy.deepcopy(self.c2)
        set_numpy_seed()

        # Test assertionError is raised with too few mating points
        with self.assertRaises(AssertionError):
            self.mate([c1, c2], n_crossover_points=0)

    def test_error_checking_too_much_crossover(self) -> None:
        c1 = copy.deepcopy(self.c1)
        c2 = copy.deepcopy(self.c2)
        set_numpy_seed()

        # Test assertionError is raised with too many mating points
        with self.assertRaises(AssertionError):
            self.mate([c1, c2], n_crossover_points=5)

    def test_error_checking_3_parent_crossover(self) -> None:
        c1 = copy.deepcopy(self.c1)
        c2 = copy.deepcopy(self.c2)
        c3 = copy.deepcopy(self.c3)
        set_numpy_seed()

        # Test assertionError is raised with too few mating points for three parents
        with self.assertRaises(AssertionError):
            self.mate([c1, c2, c3], n_crossover_points=1)

    def test_DH_crossover_same_labels(self) -> None:

        c1 = Cluster(cost=-3.5,
                     molecules=[Molecule(np.array([[0., 0, 0], [0, 4, 0]]), ["C", "H"]),
                                Molecule(np.array([[0., 0, 1], [0, 3, 0]]), ["C", "H"]),
                                Molecule(np.array([[0., 0, 2], [0, 2, 0]]), ["C", "H"]),
                                Molecule(np.array([[0., 0, 3], [0, 1, 0]]), ["C", "H"]),
                                Molecule(np.array([[0., 0, 4], [0, 0, 0]]), ["C", "H"])])

        c2 = Cluster(cost=-3.5,
                     molecules=[Molecule(np.array([[4., 0, 0], [0, 0, 4]]), ["H", "C"]),
                                Molecule(np.array([[3., 0, 1], [1, 0, 3]]), ["H", "C"]),
                                Molecule(np.array([[2., 0, 2], [2, 0, 1]]), ["H", "C"]),
                                Molecule(np.array([[1., 0, 3], [3, 0, 1]]), ["H", "C"]),
                                Molecule(np.array([[0., 2, 4], [4, 0, 0]]), ["H", "C"])])

        child = self.mate([c1, c2], n_crossover=1)

        self.log.debug(child.molecules)
        self.log.debug(child.get_molecular_positions())
        self.log.debug(child.get_particle_positions())
