# coding=utf-8
"""Threshold MC class"""
import copy
import logging
import unittest

import numpy as np

from typing import List

from bmpga.storage import Cluster, Molecule
from bmpga.mutation import BaseMutation, RandomSingleTranslation, RandomSingleRotation
from bmpga.potentials import LJcPotential

from bmpga.utils.testing_utils import set_numpy_seed
from bmpga.potentials.base_potential import BasePotential


class TestMonteCarlo(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up logging"""
        cls.log = logging.getLogger(__name__)

    def setUp(self) -> None:
        """Sets the numpy seed before each test"""
        set_numpy_seed()

    def test_MC_init(self) -> None:
        MC = MonteCarlo(potential=BasePotential(), temperature=100.0)
        self.assertTrue(isinstance(MC, MonteCarlo))

    def test_MC_init_no_temp_or_thresh(self) -> None:

        with self.assertRaises(AssertionError):
            MonteCarlo(potential=BasePotential())

    def test_MC_init_temp_and_thresh(self) -> None:

        with self.assertRaises(AssertionError):
            MonteCarlo(potential=BasePotential(), temperature=10.0, threshold=0.0)

    def test_MC_step(self):
        c1 = Cluster(molecules=[Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[1.0, 1.0, 1.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[1.0, -1.0, 1.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[-1.0, 1.0, 1.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[-1.0, -1.0, 1.0]]), particle_names=["LJ"])],
                     cost=0.0)

        pot = LJcPotential(5)
        print(pot.get_energy(c1))
        mc = MonteCarlo(potential=pot, temperature=10.0)
        mc.run(c1, 5)
        print(pot.get_energy(c1))


class MonteCarlo(object):

    def __init__(self,
                 potential: BasePotential,
                 temperature: float=None,
                 threshold: float=None,  # Not required for normal operations
                 move_classes: List[BaseMutation]=None,
                 update_steps: int=100,
                 target_acceptance=0.5,
                 log: logging.Logger=None) -> None:

        self.log = log or logging.getLogger(__name__)

        self.potential = potential
        self.T = temperature
        self.threshold = threshold
        self.target_acceptance = target_acceptance
        self.update_steps = update_steps

        self.n_accepted = 0
        self.base_factor = 0.9

        try:
            assert ((self.T is not None) or (self.threshold is not None))
        except AssertionError as error:
            self.log.exception(f"Must pass either temperature or threshold! {error}")
            raise

        self.log.debug(f"{self.T}, {self.threshold}")

        try:
            assert (self.T is None) or (self.threshold is None)
        except AssertionError as error:
            self.log.exception(f"Cannot pass both Temperature: {self.T}, and threshold: {self.threshold}")
            self.log.exception(error)
            raise

        self.move_classes = move_classes or [RandomSingleTranslation(1.0), RandomSingleRotation(2*np.pi)]

        # Remove the cluster checking.
        for _ in self.move_classes:
            _.check_cluster = lambda x: True
        self.log.info(f"Using move classes: {self.move_classes}")

        if self.threshold is not None:
            self.log.info(f"Setting up threshold MC with a threshold of: {self.threshold}")
            self.accepted = lambda new_cluster, old_cluster: new_cluster.cost <= self.threshold

    def run(self, cluster: Cluster, n_steps) -> Cluster:

        self.log.debug(f"Running MonteCarlo with T={self.T}, Threshold={self.threshold}, for {n_steps} steps")

        self.n_accepted = 0

        for step in range(n_steps):

            cluster = self.take_step(cluster)

            if step % self.update_steps == 0:
                self.update(cluster, step)
                self.log.info(f"{cluster} at step {step}")

        return cluster

    def update(self, cluster: Cluster, step: int) -> None:

        if step == 0:
            return

        print(self.n_accepted, step)
        accept_ratio: float = self.n_accepted/self.update_steps

        if accept_ratio > self.target_acceptance:
            factor = 1./self.base_factor
        elif accept_ratio < self.target_acceptance:
            factor = self.base_factor
        else:
            factor = 1.0

        self.log.info(f"""Updating step sizes at step {step}. 
        Acceptance rate = {accept_ratio}, target = {self.target_acceptance}, scaling step sizes by {factor}""")

        for move in self.move_classes:
            move.update(factor=factor)
            print(move.step_size)

        self.n_accepted = 0

    def take_step(self, old_cluster: Cluster) -> Cluster:

        new_cluster = copy.deepcopy(old_cluster)
        step = np.random.choice(self.move_classes)  # , replace=True

        new_cluster = step(new_cluster)

        new_cluster.cost = self.potential.get_energy(new_cluster)

        if self.accepted(new_cluster, old_cluster):
            self.n_accepted += 1
            return new_cluster
        else:
            return old_cluster

    def accepted(self, new_cluster: Cluster, old_cluster: Cluster):

        dU = new_cluster.cost - old_cluster.cost

        if dU < 0.0:
            return True

        p = np.random.uniform()
        p_acc = np.exp(-dU/self.T)
        return p_acc > p


if __name__ == "__main__":

    pot = LJcPotential(6)
    MC = MonteCarlo(potential=pot,
                    temperature=0.1,
                    update_steps=101,
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

    c1 = MC.run(cluster=c1, n_steps=10001)

    print(c1.cost)

