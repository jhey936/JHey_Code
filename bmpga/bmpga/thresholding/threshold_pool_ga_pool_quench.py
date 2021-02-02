# coding=utf-8
"""
Provides the threshold pool GA class
"""
import os
import time
import bmpga
import Pyro4
import shutil
import unittest
import subprocess

import numpy as np

from typing import List, Union

from bmpga.mutation import *
from bmpga.optimisation import PoolGA
from bmpga.systems import DefineSystem
from bmpga.storage import Database, Molecule, Cluster
from bmpga.characterization import SimpleEnergeticCharacterizer

from bmpga.utils.testing_utils import set_numpy_seed


class ThresholdPoolGAWithPopQuench(PoolGA):
    """ 
    Docstring to follow.
    """  # TODO Write a docstring

    def __init__(self,
                 threshold: float,
                 initial_pool: Union[List[Cluster], Database, None] = None,
                 pool_quench_freq=1,  # Default is to quench every generation
                 *args, **kwargs) -> None:

        self.threshold = threshold
        kwargs["update_freq"] = int(1e99)

        self.pool_quench_freq = pool_quench_freq

        super().__init__(*args, **kwargs)
        self.log.info(f"Running with threshold = {self.threshold}")
        self.convergence_steps = int(1e99)

        if isinstance(initial_pool, list):
            self.pool = initial_pool

        elif isinstance(initial_pool, Database):

            minima = initial_pool.get_clusters_by_cost(self.threshold)[:self.min_pool_size]
            try:
                self.pool = list(np.random.choice(minima, replace=False, size=self.min_pool_size))
            except ValueError:
                self.pool = list(np.random.choice(minima, size=self.min_pool_size))  # , replace=True

        elif initial_pool is None and self.system is not None:
            self.pool = []

        elif self.database.number_of_minima() >= self.min_pool_size:
            self.pool = list(self.database.get_clusters_by_cost()[:self.min_pool_size])

        elif initial_pool is None and self.system is None:
            try:
                raise ValueError("Initial pool is None and System is None. No way to create a valid pool.")
            except NotImplementedError as error:
                self.log.exception(error)
                raise

    def check_cluster(self, new_cluster) -> bool:
        """We keep all clusters which are more favourable than the threshold"""
        if new_cluster.cost <= self.threshold:
            return True
        else:
            return False

    def cull_pool(self) -> None:
        """We choose the new population completely stochastically"""
        new_pool = list(np.random.choice(self.pool, replace=False, size=self.min_pool_size))

        assert len(new_pool) == self.min_pool_size
        new_pool = sorted(new_pool, key=lambda x: x.cost)
        self.pool = new_pool

        generation = int(self.generations())

        if generation % self.pool_quench_freq <= 1:
            self.start_pool_quench(new_pool, generation)

        self.update()

    def start_pool_quench(self, initial_pool, generation):
        """This is a quite hacky way of getting this to work..."""
        workdir = os.path.abspath(os.curdir)

        dirname = workdir+f"/Q_Generation_{generation}"

        if not os.path.exists(dirname):
            os.mkdir(dirname)

            shutil.copy2("run_pop_quench.py", dirname)

            os.chdir(dirname)

            self.write_pool_to_file("initial_pool.xyz", pool=initial_pool)

            subprocess.Popen(['python', "run_pop_quench.py", "&"], close_fds=True)
            os.chdir(workdir)

        else:
            try:
                raise QuenchDirExistsError(f"The directory: {dirname} already exists!!")
            except QuenchDirExistsError as E:
                self.log.exception(E)
                raise


    def update(self):
        gen = int(self.generations())

        pop_ids = sorted([m.id() for m in self.pool])
        all_ens = [m.cost for m in self.pool]
        average_energy = np.mean(all_ens)

        with open(f"population", "a") as f:
            pop_str = f"{gen}, " + ", ".join([str(i) for i in pop_ids]) + "\n"
            f.write(pop_str)

        with open("avg_energy", "a") as f:
            f.write(f"{gen}, {str(average_energy)}\n")

        with open("all_en", "a") as f:
            f.write(f'{gen}, {", ".join([str(i) for i in all_ens])}\n')


class QuenchDirExistsError(Exception):
    """Exception thrown when a target quench dir already exists"""
    pass


class TestThresholdPoolGA(unittest.TestCase):
    files = []

    @classmethod
    def setUpClass(cls):
        cls.test_data_path = bmpga.__path__[0]+"/tests/test_data/"

    @classmethod
    def tearDownClass(cls):
        for file in cls.files:
            if os.path.exists(file):
                os.remove(file)

    def setUp(self):
        set_numpy_seed()

    def test_setup(self):

        nMOL = 13

        compare = SimpleEnergeticCharacterizer(accuracy=5e-7)

        db_name = f"thresh_lj{nMOL}.db"
        self.files.append(db_name)

        database = Database(db=db_name,
                            new_database=True,
                            compare_clusters=compare)

        system = DefineSystem(molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]), particle_names=["LJ"])],
                              numbers_of_molecules=[nMOL], box_length=15)

        # define a mutation scheme
        # probs is normalised inside Mutate so we can just pass realatve probabilites
        mutation = Mutate(mutations=[RandomSingleTranslation(), RandomMultipleTranslations()],
                          relative_probabilities=[1, 1])

        daemon = Pyro4.Daemon()

        ga_args = dict(database=database, min_pool_size=10, max_generations=1000, system=system, convergence_steps=500,
                       mutate=mutation, daemon=daemon, mutation_rate=0.3, max_queue_size=10,)

        thresh_GA = PoolGA(threshold=-41.326801, initial_pool=database, **ga_args)
        thresh_GA.start_threads()
        time.sleep(0.15)
        self.assertFalse(thresh_GA.jobs.empty())
        daemon.shutdown()
