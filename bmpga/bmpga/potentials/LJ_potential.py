# coding=utf-8
"""
An example of a custom potential

"""
import numpy as np
import itertools as it

import scipy.optimize

from typing import Union

from bmpga.storage import Cluster

from bmpga.potentials.base_potential import BasePotential


class GeneralizedLennardJonesPotential(BasePotential):
    """
    This is the 6-12 Generalized Lennard-Jones potential for single component systems:

    V_{LJ} = 4 \epsilon \left [ \left ( \frac{\sigma}{r} \right )^{2n} - \left ( \frac{\sigma}{r} \right )^{n} \right ]

    """
    def __init__(self, base_exponent: int = 6) -> None:
        self.epsilon4 = 4.0
        self.sigma = 1.0
        self.exp_a = int(-base_exponent / 2)  # exponent of the long-range attractive term
        self.exp_r = -base_exponent  # exponent of the short-range repulsive term
        super().__init__()

    def get_energy(self, cluster: Union[Cluster, np.ndarray], *args, **kwargs) -> float:
        """Returns the generalized lennard jones energy of the cluster

        Args:
            cluster (Cluster): required, the cluster for which to calculate energy

        Returns:
            energy (float): The energy of the cluster.

        """
        if isinstance(cluster, Cluster):
            coordinates = cluster.get_particle_positions()[0]
            n_atoms = len(coordinates)
        else:
            n_atoms = int(len(cluster)/3)
            coordinates = np.reshape(cluster, (n_atoms, 3))

        all_vecs = np.zeros((int((n_atoms * (n_atoms - 1)) / 2), 3), dtype=np.float64)

        for i, coord in enumerate(it.combinations(coordinates, 2)):
            all_vecs[i] = coord[0] - coord[1]

        rs = (all_vecs ** 2).sum(-1)
        all_e = np.sum(self.epsilon4 * (rs**self.exp_r - rs**self.exp_a))
        # noinspection PyTypeChecker
        return all_e

    def get_jacobian(self, coordinates: np.ndarray):
        """Returns a flattened array of first-order derivatives

        Args:
            coordinates: 1-d numpy.array of atomic coordinates

        Returns:
            jacobian: 1-d numpy.array of Lennard-Jones first derivatives for the system

        """

        exp_r = int(-self.exp_r*2)
        exp_a = int(-self.exp_a*2)
        n_atoms = int(len(coordinates)/3)

        coordinates = np.reshape(coordinates, newshape=(n_atoms, 3))

        all_vecs = np.zeros((int((n_atoms * (n_atoms - 1)) / 2), 3))

        for i, coord in enumerate(it.combinations(coordinates, 2)):
            all_vecs[i] = coord[1] - coord[0]

        rs = np.sqrt((all_vecs ** 2).sum(-1))

        gs = (-4.0*exp_r/rs) * (2*(1/rs)**exp_r - (1/rs)**exp_a)

        jac = np.zeros((n_atoms, 3))

        for ij, g, dr, r in zip(it.combinations(range(n_atoms), 2), gs, all_vecs, rs):
            jac[ij[0]] += -g*(dr/r)
            jac[ij[1]] += g*(dr/r)

        return jac.flatten()

    def minimize(self, cluster: Cluster, *args, **kwargs) -> dict:
        """
        Method to locally minimise a cluster of Lennard-Jones particles.
        Uses the L-BFGS-B method implemented within scipy.minimize.


        Args:
            cluster: Cluster instance, required, cluster instance to be minimized
            kwargs: Dict containing any other keyword arguments to be passed to the scipy optimizer

        Returns:
            result_dict{'coordinates': optimised structure coordinates,
                    'energy': final energy of the cluster,
                    'success': True if successfully minimised}

        """
        coordinates = cluster.get_particle_positions()
        coordinates = coordinates[0]

        result = scipy.optimize.minimize(fun=self.get_energy, x0=coordinates.flatten(),
                                         method='L-BFGS-B', jac=self.get_jacobian,
                                         **kwargs)

        return {'coordinates': result.x.reshape(coordinates.shape), 'energy': result.fun, 'success': result.success}
