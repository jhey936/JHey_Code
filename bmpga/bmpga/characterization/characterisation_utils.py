# coding=utf-8
"""Provides various methods to attempt to characterize and compare clusters to find uniqueness"""
# TODO: look here for inspiration: http://www.ccp4.ac.uk/dist//checkout/arcimboldo/src/geometry.py
import unittest

import numpy as np

from bmpga.storage import Cluster, Molecule

from bmpga.utils.geometry import find_center_of_mass


def gyration_tensor(coordinates: np.array) -> np.array:
    """Function to compute the gyration tensor of a molecule/cluster

    Args:
        coordinates: np.array(n_part, 3), required array of coordinates

    Returns:
        np.array: The 3*3 gyration tensor of the cluster as a numpy array

    """
    # Translate the cluster center of mass to the origin
    coordinates -= find_center_of_mass(coordinates, np.ones(shape=coordinates.shape[0]))

    tensor = np.zeros((3, 3))

    for x, y, z in coordinates:

        tensor[0][0] += x*x
        tensor[0][1] += x*y
        tensor[0][2] += x*z
        tensor[1][0] += y*x
        tensor[1][1] += y*y
        tensor[1][2] += y*z
        tensor[2][0] += z*x
        tensor[2][1] += z*y
        tensor[2][2] += z*z

    tensor = 1.0/float(len(coordinates)) * tensor
    # TODO: write method to rotate cluster such that \lambda_x > \lambda_y > \lambda_z
    # check to see if just sorting the diagonal gives same result?

    return tensor


def align_clusters(c1: Cluster, c2: Cluster) -> None:
    """Employs the Kabsch algorithm to align two clusters.

    c1 and c2 will be modified in-place

    See:

    Kabsch, Wolfgang, (1976) "A solution of the best rotation to relate two sets of vectors",
    Acta Crystallographica 32:922

    Args:
        c1, (Cluster): required
        c2, (Cluster): required

    Returns:
        None

    """
    # Transform both clusters CoM to the origin.
    c1.center()
    c2.center()

    coords_mols_labels_c1 = c1.get_particle_positions()
    coords_c1 = coords_mols_labels_c1[0]
    coords_mols_labels_c2 = c2.get_particle_positions()
    coords_c2 = coords_mols_labels_c2[0]

    # Calculate covarience matrix
    A = np.dot(coords_c2.transpose(), coords_c1)
    # Single value decomp
    u, s, v = np.linalg.svd(A)

    if np.linalg.det(u) * np.linalg.det(v) + 1.0 < 1e-8:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    rot_mat = np.dot(u, v).transpose()

    new_coordinates = []
    for c in coords_c2:
        new_coordinates.append(np.dot(c, rot_mat))
    new_coordinates = np.array(new_coordinates)
    c2.set_particle_positions((new_coordinates, coords_mols_labels_c2[1], coords_mols_labels_c2[2]))


class TestAlignClusters(unittest.TestCase):

    def test_simple_case(self):

        c1 = Cluster(molecules=[Molecule(coordinates=np.array([[1.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[0.0, 1.0, 0.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[0.0, 0.0, 1.0]]), particle_names=["LJ"])]
                     )
        c2 = Cluster(molecules=[Molecule(coordinates=np.array([[-1.0, 0.0, 0.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[0.0, -1.0, 0.0]]), particle_names=["LJ"]),
                                Molecule(coordinates=np.array([[0.0, 0.0, -1.0]]), particle_names=["LJ"])]
                     )

        align_clusters(c1, c2)

        for a, b in zip(c1.get_particle_positions()[0].flatten().tolist(),
                        c2.get_particle_positions()[0].flatten().tolist()):
            self.assertAlmostEqual(a, b)

        # self.assertListEqual(c1.get_particle_positions()[0].tolist(), c2.get_particle_positions()[0].tolist())


class TestGyrationTensor(unittest.TestCase):

    def test_simple_case(self):
        mol1 = Molecule(coordinates=np.array([[1.0, 0.0, 0.0], [0.0, 1., 0.0], [0.0, 0.0, 1.0],
                                              [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
                        masses=[1., 1., 1., 1., 1., 1.])

        print(gyration_tensor(mol1.coordinates, mol1.masses))
#
#     mol1.rotate(axis=np.array([0., 1., 0.]), theta=np.pi/2.)
#
#     print("-------")
#     print(gyration_tensor(mol1.coordinates, mol1.masses))
#     print("-------")
