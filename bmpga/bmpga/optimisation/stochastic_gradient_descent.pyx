# coding=utf-8
"""C implementation of the SGD"""
import numpy as np
cimport numpy as np

# from bmpga.potentials.LJ_potential_c import *

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef np.ndarray SGD_take_step(np.ndarray c1, np.ndarray all_coords):

    cdef int idx = 0
    cdef int coord_idx = 0

    cdef int npart = all_coords.size

    cdef np.ndarray new_coords = np.zeros(shape=(npart, 3), dtype=DTYPE)
    cdef np.ndarray vec = np.zeros(shape=3, dtype=DTYPE)
    # cdef np.ndarray c2 = np.zeros(shape=3, dtype=DTYPE)

    cdef np.float64 r2 = 0.0

    for idx in range(npart):
        vec = c1 - all_coords[idx]

        r2 = np.sum(vec**2)

        # ignore interation with self
        if r2 == 0.0:
            coord_idx = idx
            continue



#
# cdef class cSGD:
#
#     cpdef Cluster minimize()