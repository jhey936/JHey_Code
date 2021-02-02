# coding=utf-8
"""
Provides tests for the Genetic Algorithm (GA) classes and methods
"""
import unittest.mock
import unittest
import logging
import os

from pickle import dumps
from base64 import b64encode

from bmpga.optimisation import PoolGA
from bmpga.systems import DefineSystem
from bmpga.mating import DeavenHoCrossover
from bmpga.storage import Database, Cluster, Molecule
from bmpga.characterization import SimpleEnergeticCharacterizer

from bmpga.mating.selectors import BaseSelector
from bmpga.utils.testing_utils import set_numpy_seed, parse_info_log


class TestGA(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up a database in memory and various other classes required by PoolGA"""

        cls.log = logging.getLogger(__name__)

        cls.database = Database(new_database=True, compare_clusters=SimpleEnergeticCharacterizer())
        cls.selector = BaseSelector()
        cls.mating = DeavenHoCrossover()
        cls.mock_system = unittest.mock.Mock(spec=DefineSystem)

        cls.mock_pool = [unittest.mock.MagicMock(spec=Cluster)]*5

        cls.log.info(str(cls.mock_system))

        cls.GA = PoolGA(database=cls.database, min_pool_size=10, system=cls.mock_system,
                        crossover=cls.mating, selector=cls.selector)

        set_numpy_seed()

        def make_ga(cls, database=":memory:", min_pool_size=10,
                    selector=cls.selector, mating=cls.mating, system=cls.mock_system) -> PoolGA:
            """Returns a PoolGA instance"""

            return PoolGA(database=database, min_pool_size=min_pool_size, selector=selector,
                          crossover=mating, system=system)
        cls.make_ga = make_ga

    def test_GA_init_normal(self) -> None:

        # This should work:
        ga1 = PoolGA(self.database, min_pool_size=10, selector=self.selector,
                     crossover=self.mating, system=self.mock_system)
        self.assertIsInstance(ga1, PoolGA)

    def test_GA_init_db_string(self) -> None:

        # Check that passing a string creates a database on disk
        ga2 = PoolGA(database="test.db", min_pool_size=10, selector=self.selector,
                     crossover=self.mating, system=self.mock_system)
        # self.log.info(os.listdir("."))
        self.assertTrue(os.path.exists("test.db"))
        self.assertIsInstance(ga2, PoolGA)
        os.remove("test.db")

    def test_GA_init_bad_db(self) -> None:

        # Check AssertionError is raised when passing something not str- or Database-like.
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            PoolGA(database=123, min_pool_size=10, selector=self.selector,
                   crossover=self.mating, system=self.mock_system)

    def test_GA_init_min_max(self) -> None:

        # Check AssertionError is raised when min_ >= max_pool_size
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            PoolGA(":memory:", selector=self.selector, crossover=self.mating,
                   min_pool_size=10, max_pool_size=10, system=self.mock_system)

    def test_GA_init_bad_selector(self) -> None:

        # Check AssertionError is raised selector is not of type(bmpga.mating.BaseSelector)
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            PoolGA(":memory:", selector=self.mating, crossover=self.mating,
                   min_pool_size=10, system=self.mock_system)

    def test_GA_init_bad_crossover(self) -> None:

        # Check AssertionError is raised mating is not of type(bmpga.mating.BaseMate)
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            PoolGA(":memory:", selector=self.selector, crossover=self.selector,
                   min_pool_size=10, system=self.mock_system)

    def test_GA_get_parents(self) -> None:
        self.GA.pool = self.mock_pool
        parents = self.GA.select_clusters(n_clusters=2)
        self.log.debug(parents)
        self.assertTrue(isinstance(parents[1], Cluster))

    def test_GA_get_job(self) -> None:
        ga = self.make_ga()

        # noinspection PyTypeChecker
        ga.jobs.put(["minimise", Cluster(cost=0.0, molecules=[Molecule(coordinates=[[0., 0, 0], [1., 1, 1]],
                                                                       particle_names=["H", "LJ"])])])

        job = ga.get_job(1234)
        log_info = parse_info_log()
        job_string = b"Job assigned to worker: 1234"
        self.assertTrue(job_string in log_info)
        self.assertTrue("minimise" == job[0])


    def test_GA_run_GA_full_pool(self) -> None:

        def break_me() -> None:
            """Raises AttributeError"""
            raise AttributeError

        # noinspection PyArgumentList
        ga = self.make_ga()

        # Make pool look full
        ga.pool = [None]*5000
        ga.cull_pool = break_me

        with self.assertRaises(AttributeError):
            ga.run_GA()
        print(ga.returned_results)

    def test_GA_return_result_good_cluster(self) -> None:

        ga = self.make_ga()
        # noinspection PyTypeChecker
        self.assertIsNone(ga.return_result(
            {"data": b64encode(dumps(Cluster(cost=0.0,
                                             molecules=[Molecule(coordinates=[[0., 0, 0], [1., 1, 1]],
                                                                 particle_names=["H", "LJ"])])))}, 1234))

        result = ga.returned_results.get()

        self.assertEqual(result[0].cost, 0.0)
        self.assertEqual(result[1], 1234)
