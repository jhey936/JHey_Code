# coding=utf-8
"""Implements classes for mating"""
import copy
import logging

import numpy as np

from typing import List, Union
from itertools import cycle, combinations

from bmpga.storage import Cluster
from bmpga.utils.geometry import random_axis, magnitude
from bmpga.errors import ParentsNotSameError


class BaseMate(object):
    """Base class for mating."""
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, parents: List[Cluster], *args, **kwargs) -> Cluster:
        """Calls the self.mate() method on the supplied arguments

        Args:
            parents: List[Cluster], required, parents selected for mating.
            *args: optional, Other positional arguments
            **kwargs: optional, Other keyword arguments

        Returns:
            Cluster, the result of mating of the parent clusters (not minimised!)

        """
        return self.mate(parents, *args, **kwargs)

    @staticmethod
    def get_crossover_points(n_molecules: int, n_crossover_points: int) -> List[int]:
        """Calculates mating points when passed the number of initial_molecules
                 and number of mating points required

        Args:
            n_molecules: int, required, Number of initial_molecules in the system
            n_crossover_points: int, optional, number of mating points requested (default=1)

        Returns:
            List[int], list of points at which to perform mating

        """
        # choose mating points: between 0 and n_mol (not inclusive)
        possible_points = list(range(1, n_molecules-1))  # Creates a list of [1:n_part-1]

        #  chooses mating points from possible points. Works for single or multipoint mating
        crossover_points = np.random.choice(possible_points, size=n_crossover_points, replace=False)
        crossover_points = sorted(list(set(crossover_points)))
        crossover_points.append(None)
        return crossover_points

    def mate(self, parents: List[Cluster], *args, **kwargs) -> Cluster:
        """Takes parents and performs mating, returning an (un-minimised) child Cluster.

        Args:
            parents: List[Cluster], required, parents selected for mating.
            *args: optional, Other positional arguments
            **kwargs: optional, Other keyword arguments

        Returns:
            Cluster, the result of mating of the parent clusters (not minimised!)

        """
        raise NotImplementedError


class DeavenHoCrossover(BaseMate):
    """Implements the Deaven and Ho method for mating

    See:
    Molecular Geometry Optimization with a Genetic Algorithm,
               D.M. Deaven and K.M. Ho, Phys. Rev. Lett., 1995, 75, 288-291

    Center all of the clusters
    Apply a random rotation to each cluster
    Sort initial_molecules along the Z-axis
    Crossover at random points along the Z-axis

    Attributes:
        parents: List[Cluster], required, parents used in mating, any number of parents can be used.
        n_crossover_points: int, optional, number of mating points to use. Must be less than number of parents - 1.
                               (default=1)

    """
    def __init__(self, max_attempts: int=50, *args, **kwargs) -> None:
        self.max_attempts = max_attempts
        self.log = logging.getLogger(__name__)
        super().__init__(*args, **kwargs)

    def check_parents(self, parents: List[Cluster]) -> None:

        try:
            assert len(parents) > 1
        except AssertionError:
            message = "Too few parents passed to mating! {}".format(parents)
            self.log.exception(message)
            raise AssertionError(message)


        parent_molecules = [ sorted([mol.name for mol in parent.molecules]) for parent in parents]

        for parent in parent_molecules:
            if not parent == parent_molecules[0]:
                message = f"{parent} != {parent_molecules[0]}"
                self.log.error(message)
                raise ParentsNotSameError(message)

    def prepare_parents(self, parents: List[Cluster]) -> List[Cluster]:
        """Shuffles, centers and rotates the parent clusters

        Args:
            parents: List[Clusters], required, parents for mating.

        Returns:
            List[Clusters] shuffled, centered and rotated clusters, Note: order is not preserved.

        """
        # Shuffle the order of the parents
        np.random.shuffle(parents)

        prepared_parents = []

        # Center and then apply uniformly random rotations to each parent
        for parent in parents:
            axis = random_axis()
            parent.center()
            parent.rotate(axis, np.random.uniform(2.0 * np.pi))

            prepared_parents.append(copy.deepcopy(parent))

        return prepared_parents

    def check_child(self, parents, child_molecules: Union[List, bool]=False):
        """Checks child and parent contain the same initial_molecules by name"""
        if not child_molecules:
            return False

        child_molecule_names = sorted([mol.name for mol in child_molecules])
        parent_molecules = sorted([mol.name for mol in parents[0].molecules])

        try:
            assert parent_molecules == child_molecule_names
        except AssertionError:
            self.log.debug(f"Parent != child: {parent_molecules} != {child_molecule_names}")
            return False

        mol_coords = [m.get_center_of_mass() for m in child_molecules]
        for pair in combinations(mol_coords, 2):
            if magnitude(pair[0] - pair[1]) <= 0.6:
                self.log.debug(r"Particle overlap detected")
                return False

        return True

    def mate(self, old_parents: List[Cluster], n_crossover_points: int=1, *args, **kwargs):
        """Implements the Deaven and Ho method for mating

        See:
        Molecular Geometry Optimization with a Genetic Algorithm,
                   D.M. Deaven and K.M. Ho, Phys. Rev. Lett., 1995, 75, 288-291

        Center all of the clusters
        Apply a random rotation to each cluster
        Sort initial_molecules along the Z-axis
        Crossover at random points along the Z-axis

        Attributes:
            parents: List[Cluster], required, parents used in mating, any number of parents can be used.
            n_crossover_points: int, optional, number of mating points to use.
                                   Must be less than number of parents - 1. (default=1)

        """

        parents = [copy.deepcopy(parent) for parent in old_parents]

        self.check_parents(parents)

        try:
            assert (len(parents[0].molecules) - 1) > n_crossover_points > 0
        except AssertionError:
            message = f"Number of mating points incompatible with number of initial_molecules in system: {parents[0]}\n " \
                f"Moleules: {parents[0].molecules}\n"

            self.log.exception(message)
            raise AssertionError(message)

        try:
            assert (len(parents) - 1) <= n_crossover_points
        except AssertionError:
            message = "Number of mating points({}) incompatible with number of parents({})!" \
                .format(n_crossover_points, len(parents))
            self.log.exception(message)
            raise AssertionError(message)

        child_molecules = False

        attempts = 0

        while not self.check_child(parents, child_molecules):

            if attempts >= self.max_attempts:
                self.log.warning("Mating failed")
                return False
            else:
                attempts += 1

            self.check_parents(parents)

            new_parents = self.prepare_parents(parents)  # Validate inputs and randomise parents
            n_molecules = len(new_parents[0].molecules)

            crossover_points = set(self.get_crossover_points(n_molecules, n_crossover_points))

            child_molecules = []

            last_point = 0

            for parent, new_point in zip(cycle(new_parents), crossover_points):
                # Sort along Z axis

                parent_molecules = sorted(parent.molecules, key=lambda x: x.get_center_of_mass()[2])
                child_molecules.extend([copy.deepcopy(m) for m in parent_molecules[last_point:new_point]])
                last_point = new_point

            if len(child_molecules) >= n_molecules:
                child_molecules = child_molecules[:n_molecules]

        return Cluster(cost=0.0, molecules=copy.deepcopy(child_molecules))
