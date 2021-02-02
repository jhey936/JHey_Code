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

ClusterType = TypeVar("ClusterType", bound="Periodic")


class periodic_minimum(baseSQL):
    """
    Represents a periodic structure in the database.

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

    __tablename__ = 'tbl_periodic'

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

    def __init__(self,
                 cost: float = None,
                 molecules: List[Molecule] = None,
                 step: int = None,
                 minimum: bool = True,
                 data: List = None,
                 cluster: ClusterType = None,
                 ) -> None:
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
