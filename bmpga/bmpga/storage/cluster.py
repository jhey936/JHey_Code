# coding=utf-8
"""
Defines the main Cluster class which is the main method of storing molecular clusters

"""
import copy

import numpy as np
import itertools as it

from typing import List, Union, TypeVar, Tuple
from uuid import uuid4

from sqlalchemy.orm import deferred
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, PickleType, Boolean

from bmpga.utils.geometry import rotate, find_center_of_mass
from .molecule import Molecule


# Declare a base class
baseSQL = declarative_base()

ClusterType = TypeVar("ClusterType", bound="Cluster")


class Minimum(baseSQL):
    """Represents a structure in the database.

    Parameters:
        cost : float, value of the cost function for this cluster
        initial_molecules : list(Molecules), initial_molecules making up the cluster

    Attributes
    --------------
    cost :
        The cost of the cluster
    molecules :
        The the initial_molecules of the minimum.  This is stored as a list of Molecule entities.
    """

    __tablename__ = 'tbl_clusters'

    _id = Column(Integer, primary_key=True)

    cost = Column(Float)
    """Energy of the minimum"""
    molecules = deferred(Column(PickleType))
    """List of bmpga.storage.molecule.Molecule objects"""
    step = Column(Integer)
    """Step the minimum was found at"""
    minimum = Column(Boolean)
    data = deferred(Column(PickleType))
    """Probably an instance of a Pandas.DataFrame containing various structural data"""

    def __init__(self, cost: float=None,
                 molecules: List[Molecule]=None,
                 step=None, minimum=True, data=None,
                 cluster: ClusterType=None) -> None:
        """Initialise the cluster object

        Args:
            cost: float, required, value of the cost function for this cluster
            molecules: list(Molecules), required, list of initial_molecules in this cluster
            step: int, optional, the step at which this cluster was found
            minimum: bool, optional, set to True if this cluster is a minimum
        """
        self.public_id = self._id

        if cluster is not None:
            self.cost = copy.deepcopy(cluster.cost)
            self.molecules = copy.deepcopy(cluster.molecules)
            self.step = copy.deepcopy(cluster.step)
            self.minimum = copy.deepcopy(cluster.minimum)
            self.data = copy.deepcopy(cluster.data)
        else:
            self.cost = cost
            self.molecules = copy.deepcopy(molecules)
            self.step = step
            self.minimum = minimum
            self.data = data

    def get_id(self) -> int:
        """Returns the protected _id attribute"""
        return self._id

    def __repr__(self) -> str:
        """Returns a pretty string representation of the cluster."""
        return f"<DatabaseCluster (ID={self._id}, Energy={self.cost}, Found at step:{self.step})>"


class Cluster(object):
    """The main representation of a structure

    Parameters:
        cost : float, value of the cost function for this cluster
        initial_molecules : list(Molecules), initial_molecules making up the cluster

    Attributes
    --------------
    cost :
        The cost of the cluster
    molecules :
        The the initial_molecules of the minimum.  This is stored as a list of Molecule entities.
    """
    def __init__(self, cost: float=None,
                 molecules: List[Molecule]=None,
                 step=None, minimum=True, data=None,
                 db_cluster: Minimum=None) -> None:
        """Initialise the cluster object

        Args:
            cost: float, required, value of the cost function for this cluster
            molecules: list(Molecules), required, list of initial_molecules in this cluster
            step: int, optional, the step at which this cluster was found
            minimum: bool, optional, set to True if this cluster is a minimum
        """
        self.public_id = uuid4().int

        if db_cluster is not None:
            self.cost = copy.deepcopy(db_cluster.cost)
            self.molecules = copy.deepcopy(db_cluster.molecules)
            self.step = copy.deepcopy(db_cluster.step)
            self.minimum = copy.deepcopy(db_cluster.minimum)
            self.data = copy.deepcopy(db_cluster.data)
            self.public_id = db_cluster.get_id()
        else:
            self.cost = cost
            self.molecules = copy.deepcopy(molecules)
            self.step = step
            self.minimum = minimum
            self.data = data

        self.Natoms = None
        self.molecules = sorted(self.molecules, key=lambda x: x.name)
        self.interaction_mask = self._get_interaction_mask()

    def _get_interaction_mask(self) -> np.ndarray:
        interaction_mask = []
        for p in it.combinations(self.get_particle_positions()[1], 2):
            if p[0] == p[1]:
                interaction_mask.append(0)
            else:
                interaction_mask.append(1)
        return np.array(interaction_mask)

    def id(self) -> int:
        """Return the table id of the Cluster

        Returns:
            id: int, the unique id of this cluster within the table of clusters
        """
        return self.public_id

    def __eq__(self, other: Union[ClusterType, int]) -> bool:
        """
        Checks if cluster2 and self are the same cluster.
        Cluster can be an instance of Cluster or an id
        """
        try:
            if isinstance(other, Cluster):
                return self.id() == other.id()
            elif isinstance(other, int):
                return self.id() == other
        except TypeError:
            return False
        except AttributeError:
            return False

    def __hash__(self) -> int:
        """As the id is unique to each cluster we can use these as their hash

        Returns:
            id: int, the unique id of this cluster
        """
        return self.id()

    def __repr__(self) -> str:
        """Returns a pretty string representation of the cluster."""
        return "<Cluster (ID={}, Energy={}, Found at step:{})>".format(self.public_id, self.cost, self.step)

    def get_molecular_positions(self) -> Tuple[np.ndarray, List[int]]:
        """Returns the centers of mass of all the initial_molecules in self.initial_molecules

        Returns:
            Tuple(np.array, list[int]) coordinates of the centers of mass of all initial_molecules and their ids

        """
        coordinates, ids = [], []

        for molecule in self.molecules:
            coordinates.append(molecule.get_center_of_mass())
            ids.append(molecule.id())
        return np.array(coordinates), ids

    def get_particle_positions(self) -> Tuple[np.ndarray, List[int], List[str]]:
        """Fetches the positions of all the particles in the fragments comprising the cluster

        Returns:
            tuple(np.array, list[int], lst[str]) coordinates, id of molecule, particle label

        """

        coordinates, ids, labels = [], [], []

        for molecule in self.molecules:
            coordinates.extend(molecule.coordinates)
            ids.append(molecule.id())
            labels.extend(molecule.particle_names)

        return np.array(coordinates), ids, labels

    def set_particle_positions(self, positions: Tuple[np.ndarray, List[int], List[str]]) -> None:
        """Updated self.initial_molecules based on an input in the same
                  format as returned by self.get_particle_positions

        Args:
            positions:

        Returns:

        """

        updated_molecules = []

        if len(positions[1]) == len(positions[0]):
            # Stupid horrible hack to get a dictionary of molecule ids -> positions
            positions_dict = dict(zip(positions[1], zip(positions[0], positions[2])))



            for molecule in self.molecules:

                coordinates, labels = positions_dict[molecule.id()]

                if not isinstance(labels, list):
                    coordinates = [coordinates]
                    labels = [labels]

                molecule.coordinates = np.array(coordinates)
                molecule.particle_names = labels
                molecule.get_masses()
                updated_molecules.append(molecule)

        else:

            coordinates = positions[0]

            atom_no = 0

            for molecule in self.molecules:
                for atom, _ in enumerate(molecule.particle_names):
                    molecule.coordinates[atom] = coordinates[atom_no]
                    atom_no += 1
                updated_molecules.append(molecule)

        self.molecules = copy.deepcopy(updated_molecules)

    def rotate(self, axis: np.ndarray, theta: float) -> None:
        """Translates self.center_of_mass to origin, applies rotation then translates self back

        Args:
            axis: np.array(3), required, axis about which to rotate
            theta: float, required, angle of rotation

        Returns:
            None

        """
        center_of_mass = self.get_center_of_mass()
        self.translate(-center_of_mass)
        for molecule in self.molecules:
            molecule.coordinates = rotate(molecule.coordinates, axis, theta)
        self.translate(center_of_mass)

    def translate(self, vec: np.ndarray) -> None:
        """Translates the cluster by vec

        Args:
            vec: np.array(3), required, vector by which to translate

        Returns:
            None
        """
        for molecule in self.molecules:
            molecule.translate(vec)

    def center(self) -> None:
        """Moves the center of mass of the molecule to the origin

        Returns:
            None
        """
        self.translate(-self.get_center_of_mass())

    def get_center_of_mass(self) -> np.ndarray:
        """Returns the center of mass of the cluster

        Returns:
            np.array(3), coordinates of the center of mass

        """
        masses = np.array([mol.mass for mol in self.molecules])

        center_of_mass = find_center_of_mass(self.get_molecular_positions()[0], masses)

        return center_of_mass

    def get_natoms(self) -> int:
        """Returns the number of atoms in the cluster"""
        if self.Natoms is None:
            self.Natoms = len(self.get_particle_positions()[2])

        return self.Natoms
