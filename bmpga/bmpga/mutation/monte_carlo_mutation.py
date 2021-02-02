# coding=utf-8
"""Introduces a MonteCarlo mutation move"""
import copy

from typing import List

from bmpga.storage import Cluster
from bmpga.mutation import BaseMutation

from bmpga.potentials.base_potential import BasePotential

from bmpga.thresholding.threshold_MC import MonteCarlo


class MonteCarloMutation(BaseMutation):

    def __init__(self, potential: BasePotential,
                 move_classes: List=None,
                 threshold: float=None,
                 n_steps: int=100,
                 *args, **kwargs):

        self.n_steps = n_steps
        self.MC = MonteCarlo(potential=potential, move_classes=move_classes, threshold=threshold)
        self.check_cluster = lambda x: True

        super().__init__(*args, **kwargs)

    def mutate(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        mutated_cluster = self.MC.run(copy.deepcopy(cluster), n_steps=self.n_steps)
        return mutated_cluster


