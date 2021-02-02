# coding=utf-8
"""
Example of how to run a molecular cluster optimisation using the OPLS potential.
"""
import numpy as np

from bmpga.potentials.OPLS_ff import OPLS_potential
from bmpga.systems.define_system import DefineSystem
from bmpga.potentials.parameters.opls_parameters import GenerateOPLSMolecule
from bmpga.storage.cluster import Cluster


def create_combined_params(q, sig, eps, n_mols):
    """Combines the parameters for OPLS to get the correct format"""
    all_q, all_sig, all_eps = [], [], []

    for idx, nmol in enumerate(n_mols):
        for _ in range(nmol):
            all_q.extend(q[idx])
            all_sig.extend(sig[idx])
            all_eps.extend(eps[idx])

    return np.array(all_q), np.array(all_sig), np.array(all_eps)


tip4p_generator = GenerateOPLSMolecule("tip4p")
tip4p, tip4p_charges, tip4p_sigma, tip4p_epsilon = tip4p_generator()

system = DefineSystem(numbers_of_molecules=[2], molecules=[tip4p])


n_tip4p = 2


all_charges, all_sigma, all_epsilon = create_combined_params(q=[tip4p_charges],
                                                             sig=[tip4p_sigma],
                                                             eps=[tip4p_epsilon],
                                                             n_mols=[n_tip4p])

potential = OPLS_potential(q=all_charges, eps=all_epsilon, sigma=all_sigma)



print(potential.q)
print(potential.epsilon)
print(potential.sigma)

exit()


test_clus = system.get_random_cluster()

test_clus.molecules[0].translate(np.array([3, 3, 3]))

energy = potential.get_energy(test_clus)

print(type(test_clus))

clus2 = potential.minimize(test_clus)

print(clus2, clus2.molecules)

from bmpga.utils.io_utils import XYZWriter

writer = XYZWriter()
writer.write(clus2, "/home/john/temp/test.xyz")

print(energy)
