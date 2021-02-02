# coding=utf-8
"""
A simple pool class.
This is deprecated as it is very simplistic and is best avoided
  for use in production runs.
"""
import os
import time
import copy
import bmpga
import Pyro4
import logging

import numpy as np

from queue import Queue  # A locking fifo queue
from base64 import b64decode
from threading import Thread
from pickle import dumps, loads
# from multiprocessing import Process  # TODO: implement processes for mutation/crossover
from typing import Union, Type, List

from bmpga.systems import DefineSystem
from bmpga.storage import Database, Cluster
from bmpga.mutation import RandomClusterGenerator, Mutate
from bmpga.characterization import BaseCharacterizer, SimpleEnergeticCharacterizer

from bmpga.optimisation.base_GA import BaseGA
from bmpga.utils.io_utils import BaseWriter, XYZWriter
from bmpga.mating.mate import BaseMate, DeavenHoCrossover
from bmpga.mating.selectors import BaseSelector, RouletteWheelSelection


@Pyro4.expose
class PoolGA(BaseGA):

    # noinspection PyPep8Naming
    def __init__(self,
                 database: Union[Database, str],
                 min_pool_size: int,
                 system: Union[Cluster, DefineSystem, str, Database],
                 mutate_rate: float=0.2,  # 20% default mutation rate
                 crossover: BaseMate=DeavenHoCrossover(),
                 mutate: Mutate=Mutate(),
                 selector: BaseSelector=RouletteWheelSelection(),  # RouletteWheel by default
                 # Compare clusters based on energy:
                 compareClusters: BaseCharacterizer=SimpleEnergeticCharacterizer(1e-9),
                 writer: BaseWriter=XYZWriter(),
                 file_name: str="pool.xyz",
                 convergence_steps: int=100,
                 generations: int=100,
                 pool_unique=True,
                 max_queue_size: int=20,
                 update_freq: int=None,
                 set_initial_pool=None,
                 daemon: Pyro4.Daemon=None,
                 log: Union[Type[logging.log], None]=None,
                 max_pool_size: Union[int, None]= None,
                 *args, **kwargs) -> None:

        self.log = log or logging.getLogger(__name__)
        self.daemon = daemon

        self.database = None
        self.setup_database(database)

        self.system = self.check_system(system)

        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size or min_pool_size*2

        self.mutate_rate = mutate_rate

        self.max_generations = generations  # A generation is one*min_pool_size number of children

        self.pool_unique = pool_unique

        self.mating = crossover
        self.mutate = mutate
        self.selector = selector
        self.compareClusters = compareClusters

        self.writer = writer
        self.file_name = file_name

        self.convergence_steps = convergence_steps
        self.update_freq = update_freq or int((self.min_pool_size*self.max_generations)/10)

        super().__init__(*args, **kwargs)

        self._validate()

        self.pool = []

        self.jobs = Queue(maxsize=max_queue_size)
        self.returned_results = Queue(maxsize=max_queue_size)

        self.registered_workers = []

        self.converged = False
        self.shutdown = False
        self.n_steps = 0

        self.best_energies = []
        self.all_energies = []
        # self.__convergence_counter = 0

        if set_initial_pool is not None:
            self.set_initial_pool(set_initial_pool)

        # Setup thread to handle incoming results
        self.result_thread = Thread(target=self.process_returned_jobs, daemon=True)
        self.GA_thread = Thread(target=self.run_GA, daemon=True)

        self.start_time = time.time()

        # self.lock = Lock()
        self.processes = []

        # noinspection SpellCheckingInspection
        self.log.debug(f"Server running on Host: {os.uname()}\nWith pid: {os.getpid()}")

    def set_initial_pool(self, initial_pool: Union[List, Database, str]) -> None:
        """
        Takes an initial pool as a Database, List of Cluster objects,
        or a filename of an xyz format file and sets the initial pool

        NOT YET IMPLEMENTED
        """
        if isinstance(initial_pool, list):
            if isinstance(initial_pool[0], Cluster):
                self.pool = initial_pool
                self.log.debug(f"intial pool = {self.pool}\n{self.pool[0].molecules}")
                return
        raise NotImplementedError("This method of setting the initial pool is not yet supported")
    # noinspection PyPep8Naming
    def take_GA_step(self) -> Cluster:

        # noinspection PyArgumentEqualDefault
        if np.random.uniform(0, 1.0) >= self.mutate_rate:

            while True:
                parents = self.select_clusters(n_clusters=2)
                self.log.debug("Parents: {} & {} selected for mating".format(parents[0], parents[1]))
                child = self.mating(parents)
                self.log.debug("Child: {}".format(child))
                if isinstance(child, Cluster):
                    break
                else:
                    continue
            return child
        else:
            # Pass pool and the cluster selecting method to mutate
            mutated_cluster = self.mutate.mutate(self.select_clusters(n_clusters=1)[0])
            self.log.debug("Mutated cluster to be minimised {}".format(mutated_cluster))
            return mutated_cluster
            # self.select_clusters(n_clusters=1)

    def start_threads(self) -> None:

        self.log.debug("Starting GA thread and result thread")
        self.result_thread.start()
        self.GA_thread.start()

        self.log.info("Starting the server event loop")
        self.start_time = time.time()

    # noinspection PyPep8Naming
    def run_GA(self) -> None:
        # TODO: find some way to unittest the main run_GA method?

        self.log.info(f"""Beginning global optimisation of {self.system}. At {self.start_time}
        Using version {bmpga.__version__} of BMPGA""")

        # while max_generations is not exceeded and optimisation is not converged:
        while (self.generations() <= self.max_generations) and not self.converged and not self.shutdown:

            time.sleep(0.05)

            # If the pool is not initialised, generate and minimise a random cluster
            if len(self.pool) < self.min_pool_size:  # and not self.jobs.full():
                self.log.debug("Pool not full: {} < {}. Generating random cluster."
                               .format(len(self.pool), self.min_pool_size))

                job = dumps(["minimize", self.system.get_random_cluster()])
                # self.log.debug(job)

            # If min_pool_size < pool_size < max_pool_size, perform normal GA operation
            elif self.min_pool_size <= len(self.pool) <= self.max_pool_size:  # and not self.jobs.full():
                job = dumps(["minimize", self.take_GA_step()])

            # If the pool is too large, cull back to min_pool_size
            elif len(self.pool) >= self.max_pool_size:
                self.cull_pool()
                continue

            # If the job queue is full or another process is accessing it, wait until there
            #     is space and then put the job there.
            # Note: this will mean that the main GA will pause here until some jobs have been taken away by workers

            # noinspection PyArgumentEqualDefault,PyUnboundLocalVariable
            self.jobs.put(job, block=True, timeout=None)

        self.tear_down()

    def get_job(self, worker_id: int) -> Union[List, bytes]:
        """Method called by remote workers to retrieve jobs"""

        if self.shutdown:
            return dumps(["Shutdown", None])

        if worker_id not in self.registered_workers:
            self.registered_workers.append(worker_id)
            self.log.info("New worker {} registered".format(worker_id))

        # noinspection PyArgumentEqualDefault
        t = np.random.uniform(0.0, 1.0)
        while t < 5:

            if not self.jobs.empty():
                job = self.jobs.get()  # block=True
                self.log.debug("Job assigned to worker: {id}".format(job, id=worker_id))
                return job
            else:
                self.log.debug(f"Queue empty, waiting {t} seconds for a new job")
                time.sleep(t)
                t *= 1.1
                continue

        return dumps(["NoJob", None])

    def return_result(self, result, worker_id: int) -> None:

        result = loads(b64decode(result["data"]))

        try:
            assert isinstance(result, Cluster)
        except AttributeError as error:
            message = "Cluster object not returned by worker {}! Received: {} Type({})\n{}"
            self.log.exception(message.format(worker_id, result, type(result), error))

        result.step = self.n_steps
        result.minimum = True

        self.returned_results.put([result, worker_id])  # block=True

    def update(self) -> None:
        generation = self.generations()
        message = f"At generation {generation}. Best energy so far: {self.best_energies[-1]}"
        self.log.info(message)

        self.write_pool_to_file(filename=f"pool_generation_{generation}.xyz")

        self.log.debug(str(self.all_energies) + str(self.best_energies))

        with open("avg_energies.txt", "a+") as f:
            f.write(f"{generation},{np.mean([l[1] for l in self.all_energies[-self.min_pool_size:]])}\n")

        with open("all_energies.txt", "a+") as f:
            for line in self.all_energies:
                f.write(f"{generation},{','.join([str(x) for x in line])}\n")

        if len(self.best_energies) >= 1:
            with open("best_energies.txt", "a+") as f:
                for line in self.best_energies[1:]:
                    f.write(f"{','.join([str(x) for x in line])}\n")

        self.all_energies = []
        self.best_energies = [self.best_energies[-1]]

    def process_returned_jobs(self) -> None:

        self.log.debug("started thread in process_returned_jobs")

        while self.generations() <= self.max_generations and not self.converged and not self.shutdown:

            time.sleep(0.01)

            if not self.returned_results.empty():
                result, worker_id = self.returned_results.get()  # block=True
                self.n_steps += 1

                # Call insert first to generate cluster._id etc.
                result.step = self.n_steps

                result = self.database.insert_cluster(result)

                self.log.debug(f"Processing: {result} from worker: {worker_id}")

                if self.check_cluster(result):

                    self.pool.append(result)
                    self.log.info(f"{result} accepted. Adding to pool")

                # sort the new pool
                self.pool = sorted(self.pool, key=lambda x: x.cost)

                self.check_converged(result)

                if self.n_steps % self.update_freq == 0:
                    self.update()

        if self.converged:
            message = "Convergence reached. result_thread shutting down."
            self.log.info(message)

        elif self.generations() <= self.max_generations:
            message = f"""Max generations reached: {self.generations()} >= {self.max_generations}
            GA server will shut down shortly."""
            self.log.info(message.format)

        elif self.shutdown:
            message = "Main thread has requested shutdown. result_thread exiting."
            self.log.info(message)

        else:
            message = f"process_returned_jobs shutting down for an unknown reason!\n{self.__dict__}"
            self.log.error(message)

    def check_cluster(self, new_cluster) -> bool:  # TODO: implement uniqueness checking
        """
        Method to check the uniqueness of the returned cluster

        """
        if not self.pool_unique or len(self.pool) < 1:
            return True
        else:

            for member in self.pool:
                # This pretty much just inverts the result of compare clusters.
                # Cluster is not unique if it is the same as any other cluster
                if not self.compareClusters(new_cluster, member):
                    continue
                else:
                    return False

            self.log.debug(f"Cluster {new_cluster} unique, adding to pool")
            return True

    def check_converged(self, cluster) -> None:

        try:
            best_step, best_en = self.best_energies[-1]
        except IndexError:
            best_step, best_en = [0, 0.0]

        self.all_energies.append([self.n_steps, cluster.cost])

        if cluster.cost >= best_en:
            if (self.n_steps - best_step) >= self.convergence_steps:
                message = "Convergence reached at step: {0} (generation = {gen}). Best energy: {1} was found at step{2}"
                self.log.info(message.format(self.n_steps, best_en, best_step, gen=self.generations()))
                self.converged = True
            else:
                self.converged = False
        elif cluster.cost < best_en:
            self.best_energies.append([self.n_steps, cluster.cost])
            message = "New best energy: {} found at step: {}".format(cluster.cost, self.n_steps)
            self.log.info(message)
            self.converged = False

    def generations(self) -> float:
        """Convenience function to calculate the current generation

        Generation = current step / minimum population size
        """
        return float(float(self.n_steps)/float(self.min_pool_size))

    def cull_pool(self) -> None:
        # Reduce population back to self.min_pool_size

        # self.log.warning("GA.cull_pool not fully implemented. ")

        tmp_pool = copy.deepcopy(self.pool)

        self.pool = sorted(tmp_pool, key=lambda x: x.cost)[:self.min_pool_size]
        # raise NotImplementedError

    def select_clusters(self, n_clusters) -> List[Cluster]:
        """Invokes self.selector.select_clusters() to select parents from the pool"""
        return self.selector.get_parents(self.pool, number_of_parents=n_clusters)

    def write_pool_to_file(self, filename: str = None, pool: list = None) -> None:


        message = "Writing pool to file: {}".format(filename)
        self.log.info(message)

        if pool is None:
            pool = copy.deepcopy(self.pool)
        else:
            pool = copy.deepcopy(pool)

        self.writer.write(structures=pool, filename=filename, file_mode="w")

    def check_system(self, system: Union[Database, str, Cluster, DefineSystem])->Union[DefineSystem,
                                                                                       RandomClusterGenerator]:
        """Checks that the system passed in is

        Args:
            system: Database or str or Cluster or DefineSystem, required, an example Cluster for the system or a
                 DefineSystem object describing the system or a Database or path to a database containing at least
                 one example cluster for the system.
                 This will then be used to create an object which can produce new random clusters based on this
                 template with random molecular positions and orientations.

        Returns:
            An object with a .generate_random_cluster method for use in filling the empty pool

        Raises:
            Assertion error: if the

        """
        if isinstance(system, str) or isinstance(system, Database) or None:
            try:
                self.system = self.database.get_global_minimum()
                assert isinstance(self.system, Cluster)
            except AssertionError as error:
                message = """If passing str or database as system this must point to a valid database 
                containing at least one valid cluster. {} does not meet this requirement\n{}""".format(system, error)
                self.log.exception(message)
                raise
        else:
            try:
                assert isinstance(system, Cluster) or isinstance(system, DefineSystem)
            except AssertionError as error:
                msg = """Must pass valid database or either a description of system as either Cluster or 
                DefineSystem instance\n{}""".format(error)
                self.log.exception(msg)
                raise

            if isinstance(system, DefineSystem):
                return system
            elif isinstance(system, Cluster):
                return RandomClusterGenerator(cluster=system)

    def setup_database(self, database: Union[str, Database], *args, **kwargs) -> None:
        """Provides error checking and if everything checks out, creates a database at self.database

        Args:
            database: Union[str, Database], required, the Database object or file path to a database on disk
            *args:
            **kwargs:

        Returns:
            None. Sets self.database

        Raises:
            AssertionError when database is not valid

        """
        if isinstance(database, str):

            # noinspection PyTypeChecker
            try:
                self.database = Database(database, *args, **kwargs)  # new_database=False,

            except IOError:
                self.database = Database(database, new_database=True, *args, **kwargs)
                message = """You are using a database with the default attributes saved at: {}
                If this was not your intention you should explicitly create the database and pass it to the GA"""\
                    .format(database)
                self.log.warning(message)

        elif isinstance(database, Database):
            self.database = database

        else:

            try:
                message = "Database must be an instance of str or Database! Got {}:{}".format(type(database), database)
                raise AssertionError(message)
            except AssertionError as error:
                self.log.exception(error)
                raise

        self.log.info("Using database: {}\n".format(self.database))

    def _validate(self) -> None:  # TODO: Properly document PoolGA._validate()
        """Basic checks to ensure that that this instance of PoolGA is valid"""

        # Check the sizes of min_ and max_pool_size are valid
        try:
            assert self.min_pool_size < self.max_pool_size
        except AssertionError as error:
            self.log.exception("min_pool_size >= max_pool_size: {} >= {}.\n{}".format(self.min_pool_size,
                                                                                      self.max_pool_size,
                                                                                      error))
            raise

        try:
            assert self.mating is not None
            # noinspection PyTypeChecker
            assert isinstance(self.selector, BaseSelector)
        except AssertionError as error:
            message = "Must pass a subclass of bmpga.mating.BaseSelector as selector. Got {}:{}!\n{}"\
                .format(self.mating, type(self.mating), error)
            self.log.exception(message)
            raise

        try:
            # noinspection PyTypeChecker
            assert isinstance(self.mating, BaseMate)
        except AssertionError as error:
            message = "Must pass a subclass of bmpga.mating.BaseMate as mating. Got {}:{}!\n{}"\
                .format(self.mating, type(self.mating), error)
            self.log.exception(message)
            raise

    def tear_down(self) -> None:

        self.cull_pool()
        self.shutdown = True
        message = "GA is shutting down. Best cluster found is: {}".format(self.database.get_global_minimum())
        self.log.info(message)

        self.log.info("{pool_len} structures from pool to be written to {fn}"
                      .format(pool_len=len(self.pool), fn=self.file_name))

        self.update()
        self.write_pool_to_file(filename=self.file_name)

        self.log.info("Written best energies and all energies to disk")
        self.log.info("Shutting down daemon")
        self.daemon.shutdown()
        self.log.info("GA server was running for {}s".format(time.time() - self.start_time))
