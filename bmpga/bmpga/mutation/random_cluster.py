# coding=utf-8
"""
Mutation class to generate a new Cluster with random particle positions and orientations
"""
# import logging
import copy

import numpy as np

# from math import log as math_log
# from typing import Union

from bmpga.storage import Cluster, Molecule
from bmpga.mutation import BaseMutation

from bmpga.utils.geometry import random_axis  # , get_all_magnitudes


class RandomCluster(BaseMutation):

    def __init__(self, box_length: float=None, *args, **kwargs):

        self.box_length = box_length
        super().__init__(box_length)

    def mutate(self, old_cluster: Cluster, *args, **kwargs) -> Cluster:
        """Generates a new cluster and returns it.

        Args:
            old_cluster: Cluster object, required, The template for the cluster
            *args: unused
            **kwargs: unused

        Returns:
            A new cluster with new molecular positions and orientations

        """
        max_step = self.box_length or len(old_cluster.molecules)**(5./4)

        # new_cluster = copy.deepcopy(old_cluster)
        #
        # base_molecules = copy.deepcopy(old_cluster.molecules)
        #
        # new_cluster.molecules = []
        #
        # for m in base_molecules:
        #     m.center()
        #     vec = np.random.uniform(low=0.1, high=max_step, size=3)
        #     m.translate(vec)
        #
        #     m.rotate(random_axis(), np.random.uniform(0, 2 * np.pi))
        #
        #     new_cluster.molecules.append(m)
        #
        # return new_cluster

        new_molecules = [Molecule(coordinates=copy.deepcopy(m.coordinates), particle_names=m.particle_names)
                         for m in old_cluster.molecules]

        for m in new_molecules:

            m.center()

            vec = np.random.uniform(low=0.1, high=max_step, size=3)
            m.translate(vec)

            m.rotate(random_axis(), np.random.uniform(0, 2 * np.pi))

        new_cluster = Cluster(molecules=new_molecules)
        return new_cluster


class RandomClusterGenerator(RandomCluster):
    """Simple class to provide an interface to RandomCluster"""
    def __init__(self, cluster: Cluster=None, *args, **kwargs):

        self.template = cluster
        super().__init__(*args, **kwargs)

    def get_random_cluster(self) -> Cluster:
        """Returns a random cluster based on the template"""
        return self.mutate(self.template)
