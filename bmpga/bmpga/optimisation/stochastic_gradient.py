# coding=utf-8
"""Stochastic Gradient Descent"""  # TODO: Write C++ implementation
import copy

import numpy as np

from bmpga.storage import Cluster
from bmpga.potentials import LJcPotential


class LJcStochastic(LJcPotential):
    """Potential for testing"""

    def __init__(self, n):
        super().__init__(n)

    def _calc_g(self, rsq):

        ir2 = 1./rsq
        ir6 = ir2**3
        ir12 = ir6 ** 2

        g = -4.0 * ir2 * ((12.0 * ir12) - (6.0 * ir6))
        return g


class SGD(object):

    def __init__(self,
                 n_particles=None,
                 potential=None,
                 convergence_energy_gradient: float=1e-6,
                 initial_step: float=0.01,
                 max_stpes: int=10000):

        self.pot = potential or LJcStochastic(n_particles)
        self.convergence_gradient = convergence_energy_gradient

        self.initial_step = initial_step
        self.max_steps = max_stpes

    def __call__(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        return self.minimize(cluster)

    def minimize(self, cluster: Cluster) -> Cluster:

        self.step_size = self.initial_step
        step = 0

        _ = cluster.get_particle_positions()
        # n_coords = len(_[1])
        self.coords = _[0]

        np.random.shuffle(self.coords)

        last_energy = self.pot.get_energy(cluster)

        converged = False
        while step <= self.max_steps and not converged:
            converged = True
            step += 1

            for idx in range(len(self.coords)):
                self._take_step(self.coords[idx])

            cluster.set_particle_positions((self.coords, _[1], _[2]))
            new_energy = self.pot.get_energy(cluster)
            # print(new_energy, last_energy)
            dE = new_energy - last_energy
            last_energy = new_energy
            self.update()

            if abs(dE) >= self.convergence_gradient:
                converged = False

        # print(f"Exiting in {step} steps")
        # for i in self.coords:
        #     print("Cl   " + "   ".join([str(a) for a in i]))
        cluster.minimum = True
        return cluster

    def update(self) -> None:
        np.random.shuffle(self.coords)

    def _take_step(self, c1) -> None:

        new_coords = []

        c1_new_coords = copy.deepcopy(c1)

        coord_index = int(1e9)

        for idx, c2 in enumerate(self.coords):
            vec = c1 - c2
            r = (vec**2).sum(-1)

            # Deal with when c1 == c2
            if r <= 0.2:
                coord_index = idx
                new_coords.append(c2)
                continue

            g = self.pot._calc_g(r)

            c1_new_coords -= self.step_size * vec * g
            c2_new_coords = c2+self.step_size * vec * g
            new_coords.append(c2_new_coords)

        new_coords = np.array(new_coords)
        new_coords[coord_index] = c1_new_coords
        self.coords = new_coords
