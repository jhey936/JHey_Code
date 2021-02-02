# coding=utf-8
"""Provides an interface to functions written in C to accelerate the calculation of the Lennard-Jones potential"""
import logging

import scipy.optimize

import numpy as np

from typing import Union

# noinspection PyPep8
import pyximport; pyximport.install(setup_args={"include_dirs": [np.get_include()]})

from bmpga.potentials.base_potential import BasePotential
from bmpga.potentials.LJ_potential_c import c_get_energy, c_get_jacobian, c_get_g
from bmpga.storage import Cluster


class LJcPotential(BasePotential):
    """Provides an interface to the C implementation of the Lennard-Jones potential"""
    def __init__(self, n_atoms,
                 sigma: Union[np.ndarray, float] = 1.0,
                 epsilon: Union[np.ndarray, float] = 1.0,
                 base_exp: int = 6,
                 interaction_mask: np.ndarray = None,
                 log=None,
                 *args, **kwargs) -> None:

        self.log = log or logging.getLogger(__name__)

        self.n_atoms = n_atoms
        self.n_interactions = int((n_atoms * (n_atoms - 1)) / 2)

        # Used to remove self-interaction (elements set to zero remove interaction)
        if interaction_mask is None:
            self.interaction_mask = np.ones(self.n_interactions, dtype=np.float64)
        else:
            self.interaction_mask = interaction_mask

        self.base_exp = base_exp

        if isinstance(sigma, float):
            sigma = [sigma]*self.n_interactions
            self.sigma = np.array(sigma, dtype=np.float64)
        elif isinstance(sigma, np.ndarray):
            self.sigma = sigma
            assert self.sigma.dtype == np.float64
        else:
            try:
                raise TypeError(f"Expecting np.array or float received: {sigma}, {type(sigma)}\n")
            except TypeError as err:
                self.log.exception(err)
                raise

        if isinstance(epsilon, float):
            epsilon = [epsilon]*self.n_interactions
            self.epsilon4 = 4*np.array(epsilon, dtype=np.float64)
        elif isinstance(epsilon, np.ndarray):
            self.epsilon4 = epsilon*4.0
            assert self.epsilon4.dtype == np.float64
        else:
            try:
                raise TypeError(f"Expecting np.array or float received: {epsilon}, {type(epsilon)}\n")
            except TypeError as err:
                self.log.exception(err)
                raise

        try:
            assert self.sigma.size == self.epsilon4.size
            assert self.sigma.size == int((self.n_atoms * (self.n_atoms - 1)) / 2)
        except AssertionError as err:
            self.log.exception(f"""sigma and epsilon and n_atoms must both be of length (n_atoms*(n_atoms-1))/2!
            sigma={sigma}, epsilon={epsilon}\n{err}""")

        super().__init__(*args, **kwargs)

    def get_energy(self, cluster: Union[np.ndarray, Cluster], **kwargs) -> float:
        """Returns the full system interaction energy from the C implementation of the Lennard-Jones potential

        Args:
            cluster:
            **kwargs:
            **kwargs: dict, optional, extra keyword arguments

        Returns:
            U_{LJ}: float, the Lennard-Jones interaction energy

        """
        if isinstance(cluster, Cluster):
            positions = cluster.get_particle_positions()
            coordinates = positions[0]
            self.interaction_mask = cluster.interaction_mask

        elif isinstance(cluster, np.ndarray):
            coordinates = np.reshape(cluster, (int(len(cluster)/3), 3))

        else:
            try:
                raise TypeError(f"Expecting Cluster or np.ndarray received: {cluster}, {type(cluster)}")
            except TypeError as err:
                self.log.exception(err)
                raise

        assert coordinates.dtype == np.float64

        return c_get_energy(coordinates, self.sigma, self.epsilon4, self.interaction_mask, self.base_exp)

    def get_jacobian(self, coordinates: np.ndarray) -> np.array:
        """Constructs the jacobian for for Lennard-Jones system

        Args:
            coordinates: np.array, required, coordinate array

        Returns:
            np.array of first derivatives

        """
        return c_get_jacobian(coordinates, self.sigma, self.epsilon4, self.interaction_mask, self.base_exp)

    def minimize(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """
        Method to locally minimise a cluster of Lennard-Jones particles.
        Uses the L-BFGS-B method implemented within scipy.minimize.

        Attributes:
            coordinates: np.array(shape=(number_of_particles, 3), dtype=float) array of coordinates
            kwargs: Dict containing any other keyword arguments to be passed to the scipy optimizer
            molecules: list(int), optional,

        Returns
        -------
        result_dict{'coordinates': optimised structure coordinates,
                    'energy': final energy of the cluster,
                    'success': True if successfully minimised}

        """
        positions = list(cluster.get_particle_positions())
        coordinates = positions[0].flatten()

        # args = {"sigma": self.sigma, "epsilon": self.epsilon4, "base_exp": 6}

        result = scipy.optimize.minimize(fun=self.get_energy, x0=coordinates,  # , args=args,
                                         method='L-BFGS-B', jac=self.get_jacobian)

        positions = (result.x.reshape((self.n_atoms, 3)), positions[1], positions[2])
        cluster.set_particle_positions(positions)
        cluster.cost = result.fun
        return cluster

    def _calc_g(self, rsq):
        return c_get_g(rsq, 1.0, 4.0)
