# coding=utf-8
"""Threshold MC class"""
import logging
import unittest

import numpy as np

from collections import Counter
from multiprocessing import Pool

from bmpga.storage import Cluster, Molecule, Database
from bmpga.mutation import RandomSingleTranslation
from bmpga.potentials import LJcPotential

from bmpga.utils.testing_utils import set_numpy_seed
from bmpga.thresholding.threshold_MC import MonteCarlo
from bmpga.optimisation.stochastic_gradient import SGD
from bmpga.potentials.base_potential import BasePotential
from bmpga.characterization.simple_characterizer import SimpleGeometricCharacterizer


class TestMonteCarloStochasticQuench(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up logging"""
        cls.log = logging.getLogger(__name__)

    def setUp(self) -> None:
        """Sets the numpy seed before each test"""
        set_numpy_seed()

    def test_MC_init(self) -> None:
        MC = MonteCarloStochasticQuench(potential=BasePotential(), temperature=100.0)
        self.assertTrue(isinstance(MC, MonteCarlo))

    def test_MC_init_no_temp_or_thresh(self) -> None:

        with self.assertRaises(AssertionError):
            MonteCarloStochasticQuench(potential=BasePotential())

    def test_MC_init_temp_and_thresh(self) -> None:

        with self.assertRaises(AssertionError):
            MonteCarloStochasticQuench(potential=BasePotential(), temperature=10.0, threshold=0.0)

    def test_MC_step(self):
        c1 = Cluster(molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[1.0, 1.0, 1.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[1.0, -1.0, 1.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[-1.0, 1.0, 1.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[-1.0, -1.0, 1.0]]), particle_names=["LJ"])],
                     cost=0.0)

        pot = LJcPotential(5)
        print(pot.get_energy(c1))
        mc = MonteCarloStochasticQuench(potential=pot, temperature=10.0)
        mc.run(c1, 5)
        print(pot.get_energy(c1))


class MonteCarloStochasticQuench(MonteCarlo):
    """Inherits from MonteCarlo and implements stochastic quenching.
    Every update_steps steps, n_quench stochastic quenches are performed from the current minimum.
    The results of these are inserted into the database.

    """

    def __init__(self,
                 n_quench: int=100,
                 database: Database=Database(new_database=True, compare_clusters=SimpleGeometricCharacterizer()),
                 n_processes: int=2,
                 *args, **kwargs) -> None:

        self.n_quench = n_quench
        self.database = database
        self.n_processes = n_processes

        self.quench = None
        self.results_dict = {}

        super().__init__(*args, **kwargs)

    def run(self, cluster: Cluster, n_steps) -> dict:

        # setup quench here, so we know how many atoms we are dealing with...
        self.quench = SGD(len(cluster.get_particle_positions()[0]))
        super().run(cluster, n_steps)
        return self.results_dict

    def update(self, cluster: Cluster, step: int) -> None:
        super().update(cluster=cluster, step=step)

        cluster.minimum = False
        cluster = self.database.insert_cluster(cluster)

        worker_pool = Pool(processes=self.n_processes)

        minima = Pool.map(self.quench(cluster), [cluster for i in range(self.n_quench)])

        for _ in range(self.n_quench):

            minimum = self.quench(cluster)
            minimum.step = step
            minimum = self.database.insert_cluster(minimum)

            minima.append(minimum)

        self.results_dict[cluster] = Counter(minima)


if __name__ == "__main__":

    pot = LJcPotential(6)
    MC = MonteCarloStochasticQuench(
        potential=pot,
        n_quench=1,
        threshold=-1.0,
        update_steps=10,
        move_classes=[RandomSingleTranslation()])

    c1 = Cluster(molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]), particle_names=["LJ"]),
                            Molecule(coordinates=np.array([[1.0, 1.0, 1.0]]), particle_names=["LJ"]),
                            Molecule(coordinates=np.array([[1.0, -1.0, 1.0]]), particle_names=["LJ"]),
                            Molecule(coordinates=np.array([[-1.0, 1.0, 1.0]]), particle_names=["LJ"]),
                            Molecule(coordinates=np.array([[-1.0, -1.0, 1.0]]), particle_names=["LJ"]),
                            Molecule(coordinates=np.array([[-2.0, -2.0, 2.0]]), particle_names=["LJ"])],
                 cost=0.0)
    c1 = pot.minimize(cluster=c1)
    c1.cost = pot.get_energy(c1)
    print(c1.cost)
    print(MC.move_classes[0].random)

    c1 = MC.run(cluster=c1, n_steps=100)

    print(c1)

