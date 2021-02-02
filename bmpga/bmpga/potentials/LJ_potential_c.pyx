# coding=utf-8
"""Example of a custom potential implemented using cython"""

# noinspection PyUnresolvedReferences
cimport numpy as np
import numpy as np
import itertools as it

# noinspection PyPep8Naming
DTYPE = np.float64
ctypedef np.double_t  DTYPE_t


cpdef c_get_g(double r2,
              double sigma=1.0,
              double epsilon4=4.0,
              ):
    """Returns pairwise gradients"""

    cdef DTYPE_t ir2
    cdef DTYPE_t ir6
    cdef DTYPE_t ir12

    cdef DTYPE_t sig6
    cdef DTYPE_t sig12

    ir2 = 1./r2
    ir6 = ir2**3
    ir12 = ir6**2

    sig6 = sigma**6
    sig12 = sig6**2

    return -epsilon4 * ir2 * ((12.0*sig12*ir12) - (6.0 * sig6 * ir6))

# noinspection PyMissingTypeHints
cpdef c_get_jacobian(np.ndarray coordinates,
                     np.ndarray sigma,
                     np.ndarray epsilon4,
                     np.ndarray interaction_mask,
                     int base_exp=6,
                     ):
    """Calculates the jacobian for the given coordinates

    Returns a flattened array of first-order derivatives for
    the Lennard-Jones potential using calls to methods implemented
    in C

    Args:
        interaction_mask: 
        coordinates: 1-d numpy array, array of particle coordinates
        sigma: double, the Lennard-Jones sigma parameter (default=1)
        epsilon4: double, 4*the Lennard-Jones epsilon parameter (default=1)
        base_exp: The base exponent parameter for a 2n-n lennard jones potential

    Returns:
        jacobian: 1-d array of first derivatives for the system

    """
    cdef Py_ssize_t number_of_atoms = int(len(coordinates)/3)
    cdef Py_ssize_t number_of_interactions = int((number_of_atoms*(number_of_atoms-1))/2)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0

    cdef int exp_r = 2*base_exp
    cdef int exp_a = base_exp

    coordinates = np.reshape(coordinates, newshape=(number_of_atoms, 3))

    cdef np.ndarray jacobian = np.zeros(dtype=DTYPE, shape=(number_of_atoms, 3))
    cdef np.ndarray dr = np.zeros(3, dtype=DTYPE)
    cdef DTYPE_t r2
    cdef DTYPE_t ir2
    cdef DTYPE_t ir6
    cdef DTYPE_t ir12

    cdef DTYPE_t sig6
    cdef DTYPE_t sig12

    cdef DTYPE_t g

    for i in range(number_of_atoms):
        for j in range(i+1, number_of_atoms):

            dr = coordinates[j] - coordinates[i]

            r2 = (dr**2).sum(-1)

            ir2 = 1./r2
            ir6 = ir2**3
            ir12 = ir6**2

            sig6 = sigma[i]**6
            sig12 = sig6**2

            g = -epsilon4[i] * ir2 * ((12.0*sig12*ir12) - (6.0 * sig6 * ir6))

            jacobian[i] += -g*dr
            jacobian[j] += g*dr

    return jacobian.flatten()

# noinspection PyMissingTypeHints
cpdef c_get_energy(
        np.ndarray coordinates,
        np.ndarray sigma,
        np.ndarray epsilon4,
        np.ndarray interaction_mask,
        int base_exp=6,
        ):
    """Returns the total Lennard-Jones potential energy at a given set of coordinates
    
    Args:
        interaction_mask:
        coordinates: np.array(n_atoms, 3), required, coordinate array
        sigma: float, optional, Lennard-Jones sigma parameter
        epsilon4: float, optional, 4*Lennard-Jones epsilon parameter (default=4)
        base_exp: int, optional, the base exponent for the 2n-n Lennard Jones potential (default-6)
        
    Returns:
        U_lj: float, interaction energy

    """
    cdef int number_of_atoms = int(len(coordinates))
    cdef Py_ssize_t number_of_interactions = int((number_of_atoms*(number_of_atoms-1))/2)

    cdef Py_ssize_t i = 0

    cdef int exp_r = -base_exp
    cdef int exp_a = -base_exp/2

    cdef DTYPE_t energy = 0.0

    all_vecs = np.zeros(shape=(number_of_interactions, 3), dtype=DTYPE)
    all_rsq = np.zeros(shape=number_of_interactions, dtype=DTYPE)
    v1 = np.zeros(3, dtype=DTYPE)
    v2 = np.zeros(3, dtype=DTYPE)

    i = 0
    for v1, v2 in it.combinations(coordinates, 2):
        all_vecs[i] = v1-v2
        i += 1

    all_rsq = (all_vecs**2).sum(-1)

    energy = np.sum(epsilon4 * ((all_rsq**exp_r) - (all_rsq**exp_a)) * interaction_mask)

    return energy
