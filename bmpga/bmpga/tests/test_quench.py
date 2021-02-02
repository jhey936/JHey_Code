# coding=utf-8
"""
Provides tests for the local optimisation classes and methods


    NOTE: As this file sets up the server in a separate python thread,
            you must have this file running as main for the tests to pass.

            So in one terminal run:
            $ python test_quench.py

            This starts the test server using __name__ == "__main__": ...

            then in a separate terminal:
            $ python -m unittest test_quench.py

"""
import os
import copy
import bmpga
import Pyro4
import logging
import unittest
import subprocess

import numpy as np

from pickle import loads, dumps
from base64 import b64decode, b64encode

from bmpga.storage import Cluster
from bmpga.errors import ClientError
from bmpga.potentials import LJcPotential
from bmpga.optimisation import QuenchClient

from bmpga.storage.molecule import Molecule


@Pyro4.expose
class DummyGA(object):
    """Dummy GA for testing the quench client"""
    def __init__(self, max_quenches: int=10) -> None:

        self.max_quenches = max_quenches
        self.total_quenches = 0

        self.calls = 0
        self.cluster = Cluster(cost=0.0, molecules=[Molecule(coordinates=np.array([[0., 0., 0.], [1., 1., 1.]]),
                                                             particle_names=["LJ", "LJ"])])
        self.log = logging.getLogger(__name__)
        self.log.debug("Started DummyGA")
        self.all_jobs = [dumps(["minimize", copy.deepcopy(self.cluster)]),
                         dumps(["energy", copy.deepcopy(self.cluster)]),
                         dumps(["random", copy.deepcopy(self.cluster)])]*5

    def get_job(self, _id) -> bytes:
        """Returns a list in the expected format for self.get_job()"""

        self.log.debug("DummyGA giving job number: {} to quencher: {}".format(self.calls, _id))

        job = self.all_jobs[self.calls]

        # for job in all_jobs:
        self.log.debug("DummyGA giving job: {}: {}".format(self.calls, job))
        self.calls += 1
        return job

    def return_result(self, returned_data: dict, _id) -> bool:
        """Accepts a returned_data object"""
        self.log.info("Received: {} from: ".format(loads(b64decode(returned_data["data"])), _id))
        self.total_quenches += 1
        return self.total_quenches < self.max_quenches


class TestDummyGA(unittest.TestCase):
    """Decided to test the dummy server for some reason."""
    @classmethod
    def setUpClass(cls) -> None:
        """Setup class-wide logging"""
        cls.log = logging.getLogger(__name__)

    def test_init_dummy_ga(self) -> None:
        self.assertIsInstance(DummyGA(), DummyGA)

    def test_init_dummy_ga_max_job_number(self) -> None:
        self.assertIsInstance(DummyGA(max_quenches=1), DummyGA)

    def test_dummy_ga_get_job(self) -> None:
        ga = DummyGA()
        job = ga.get_job(321)
        self.assertTrue(b"bmpga.storage.cluster" in job)

    def test_dummy_ga_return_job_max_jobs_reached(self) -> None:
        ga = DummyGA(max_quenches=1)
        self.assertFalse(ga.return_result({"data": b64encode(dumps(1234))}, 321))

    def test_dummy_ga_return_job_max_jobs_not_reached(self) -> None:
        ga = DummyGA(max_quenches=3)
        self.assertTrue(ga.return_result({"data": b64encode(dumps(1234))}, 321))


class TestQuench(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the quencher and a DummyGA server for testing"""
    
        cls.log = logging.getLogger(__name__)
        command = "python3 {}/tests/test_quench.py &".format(bmpga.__path__[0])
        # cls.running_subprocess = subprocess.call(command, shell=True)
        cls.running_subprocess = subprocess.Popen(command, shell=True)
        cls.log.info(cls.running_subprocess)
        cls.uri = "test_uri.log"

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.log.debug(cls.running_subprocess)
        os.kill(cls.running_subprocess.pid, 9)
        cls.running_subprocess.kill()
        os.remove(cls.uri)

    def test_a_quench_init(self) -> None:
        potential = LJcPotential(1)
        # noinspection PyTypeChecker
        quencher = QuenchClient(potential, URI=self.uri)
        self.assertIsInstance(quencher, QuenchClient)

        # Check that providing no URI raises the correct error
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            QuenchClient(potential)

        # Check that uri_from_file works
        uri = quencher.uri_from_file(self.uri)
        self.assertIsInstance(uri, str)

    def test_quench_run(self) -> None:

        self.log.info("Starting quench client")
        potential = LJcPotential(2)
        # noinspection PyTypeCheckers
        quencher = QuenchClient(potential, URI=self.uri, max_quenches=1)
        with self.assertRaises(SystemExit):
            quencher.run()

    def test_quench_run_bad(self) -> None:
        potential = LJcPotential(2)
        self.log.info("Starting bad quench client")
        # noinspection PyTypeChecker
        quencher = QuenchClient(potential, URI=self.uri, max_quenches=7)
        # This is expected to fail because the server is set up to pass a
        #     bad job description to the quencher after some time
        with self.assertRaises(ClientError):
            quencher.run()


if __name__ == "__main__":
    Pyro4.config.COMPRESSION = False
    # noinspection SpellCheckingInspection
    Pyro4.config.SERVERTYPE = "thread"
    Pyro4.config.SERIALIZER = "pickle"

    daemon = Pyro4.Daemon()
    with open("test_uri.log", "w") as f:
        f.write(str(daemon.register(DummyGA())))

    daemon.requestLoop()
