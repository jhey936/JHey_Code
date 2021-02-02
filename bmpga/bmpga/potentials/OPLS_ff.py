# coding=utf-8
"""
Base potential class.
New potentials should inherit from here.
See lennard_jones.py for an example of an implemented class
"""
import numpy as np
import itertools as it

from typing import Union

from scipy.optimize import minimize

from bmpga.storage import Cluster
from bmpga.utils.geometry import get_all_magnitudes
from bmpga.potentials.base_potential import BasePotential


# noinspection PyPep8Naming
class OPLS_potential(BasePotential):
    """The base object new potentials should inherit from.

    This potential assumes the ordering of the atoms does not change during the optimisation.

    Methods:
        get_energy: Return the energy of the system at the current time
        minimize: minimize and return coordinates and energy of the system

    """

    def __init__(self, q, eps, sigma, interaction_mask=None, *args, **kwargs) -> None:

        self.check_inputs(q, eps, sigma, interaction_mask)

        self.q, self.epsilon, self.sigma = self.calculate_cross_terms(q, eps, sigma, interaction_mask)


        super().__init__(*args, **kwargs)

    @staticmethod
    def check_inputs(q, eps, sigma, interaction_mask):
        """Checks the inputs for compatable lengths etc."""
        if len(q) != len(eps) or len(q) != len(sigma) or len(eps) != len(sigma):

            message = f"""
Charges, sigma and/or epsilon lists are not the same lenght! 
len(q) = {len(q)}            
len(epsilon) = {len(eps)}            
len(sigma) = {len(sigma)}            
"""
            raise AttributeError(message)

        if interaction_mask is not None:
            if isinstance(interaction_mask[0], np.bool) and \
                isinstance(interaction_mask[0], np.int) and \
                isinstance(interaction_mask[0], np.float64) or\
                    not isinstance(interaction_mask, np.ndarray):

                message = f"""
Interaction mask must be a np.ndarray of with datatype == np.bool, np.int or np.float64
Type(interaction_mask) = {type(interaction_mask)}
Datatype = {type(interaction_mask[0])}
"""
                raise AttributeError(message)

            expected_interaction_mask_len = int((len(q) * (len(q) - 1)) / 2)
            if len(interaction_mask) != expected_interaction_mask_len:
                    message = f"""
interaction_mask expected to be of length: {expected_interaction_mask_len}
not! : {len(interaction_mask)}
            """
                    raise AttributeError(message)

            if max(interaction_mask) > 1.0 or max(interaction_mask) < 1.0:
                message = f"max value of interaction mask must be 1. Not: {max(interaction_mask)}"
                raise AttributeError(message)

    @staticmethod
    def calculate_cross_terms(q_old: Union[np.array, list],
                              eps_old: Union[np.array, list],
                              sigma_old: Union[np.array, list],
                              interaction_mask: Union[np.array] = None) -> Union[np.array]:

        """Calculates and returns cross terms for the LJ and coulombic interaction terms.

        Args:
            q_old: list of charges
            eps_old: list of epsilon terms
            sigma_old: list of sigma terms
            interaction_mask: mask for interactions. Boolian or binary np.array, default = None

        Returns:
            q, epsilon, sigma as a tuple of np.arrays

        """
        arr_len = int((len(q_old) * (len(q_old)-1))/2)

        q = np.zeros(shape=arr_len)
        epsilon = np.zeros(shape=arr_len)
        sigma = np.zeros(shape=arr_len)

        idx = 0

        for i, _ in enumerate(q_old):

            for j in range(1+i, len(q_old)):

                q[idx] = q_old[i] * q_old[j]
                sigma[idx] = (sigma_old[i] + sigma_old[j])/2
                epsilon[idx] = (eps_old[i] + eps_old[j])/2
                idx += 1

        if interaction_mask is not None:
            q = q * interaction_mask
            epsilon = epsilon * interaction_mask
            sigma = sigma * interaction_mask

        return q, epsilon, sigma

    def get_energy(self, cluster: Cluster) -> float:
        """
        Method to return the single-point energy of a system

        Parameters
        ----------


        Returns
        -------
        float(energy): Energy of the system at the given coordinates

        Args:
            cluster: Cluster instance, required, the for which to calculate energy
            *args: list, optional, postitional arguments
            **kwargs: Other keyword arguments needed on a per-implementation basis (i.e. atom labels)

        """
        if isinstance(cluster, Cluster):
            coordinates, _, _ = cluster.get_particle_positions()
        elif isinstance(cluster, np.ndarray):
            natoms = int(len(cluster)/3)
            coordinates = cluster.reshape((natoms, 3))
        else:
            raise AttributeError("coordinate format is not accepted")

        rs = get_all_magnitudes(coordinates)
        coulombic = self.coulombic_E_component(rs)

        LJ_component = self.LJ_E_component(rs)
        return coulombic + LJ_component

    def LJ_E_component(self, rs: np.array) -> float:
        """Return the lennard-jones energy component"""
        sig_r = (self.sigma/rs)**6

        return sum(4*self.epsilon*(sig_r**2 - sig_r))

    def coulombic_E_component(self, rs: np.array) -> float:
        """Calculates the coulombic component of the OPLS ff"""
        return sum(self.q/rs)

    # noinspection PyMethodMayBeStatic
    def get_jacobian(self, coordinates: np.ndarray):
        """Returns a flattened array of first-order derivatives

        Args:
            coordinates: 1-d numpy.array of atomic coordinates

        Returns:
            jacobian: 1-d numpy.array of Lennard-Jones first derivatives for the system

        """

        n_atoms = int(len(coordinates)/3)
        coordinates = np.reshape(coordinates, newshape=(n_atoms, 3))

        all_vecs = np.zeros((int((n_atoms * (n_atoms - 1)) / 2), 3))

        for i, coord in enumerate(it.combinations(coordinates, 2)):
            all_vecs[i] = coord[1] - coord[0]

        rs = np.sqrt((all_vecs ** 2).sum(-1))

        gs = ((24*self.epsilon*(self.sigma**6) * ((rs**6)-(2*(self.sigma**6))))/(rs**13))-(self.q/(rs**2))

        jac = np.zeros((n_atoms, 3))

        for ij, g, dr, r in zip(it.combinations(range(n_atoms), 2), gs, all_vecs, rs):

            jac[ij[0]] -= g*dr/r
            jac[ij[1]] += g*dr/r

        return jac.flatten()

    def minimize(self, cluster: Cluster, *args, **kwargs) -> Cluster:
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
        coords, ids, labels = cluster.get_particle_positions()
        coordinates = coords

        result = minimize(fun=self.get_energy, x0=coordinates.flatten(),
                          method='L-BFGS-B',
                          jac=self.get_jacobian,
                          *args, **kwargs)

        if not result.success:
            print("Optimization failed")

        cluster.set_particle_positions((result.x.reshape(coordinates.shape), ids, labels))
        cluster.cost = result.fun
        return cluster
