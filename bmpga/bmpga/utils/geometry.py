# coding=utf-8
"""Provides various utilities for manipulating geometries"""
import logging

import itertools as it
import numpy as np

from bmpga.utils.chem import get_masses

log = logging.getLogger(__name__)


def random_axis(minimum: int=-10, maximum: int=10) -> np.ndarray:
    """Returns an integer vector of length 3

    Uses numpy.random.randint which draws uniform integers in the half-open interval [minimum, maximum)

    Args:
        minimum: int, optional, Minimum integer value (inclusive) (default=0)
        maximum: int, optional, Maximum integer value (exclusive) (default=10)

    Returns:
        np.array: Length 3 vector of uniformly random ints

    """
    return np.random.randint(minimum, maximum, size=3)


def get_rotation_matrix(axis: np.ndarray, theta: float=0.0) -> np.ndarray:
    """
    Formulates a rotation matrix for the rotation about the supplied axis

    Parameters
    ----------
    axis = axis about which to rotate
    theta = angle of rotation

    Returns
    -------
    3*3 rotation matrix for rotation about axis theta

    """
    axis = np.array(axis)
    if np.array_equal(np.array([0.0, 0.0, 0.0]), axis):
        log.warning("Warning: Axis == {}\nRotating about x instead\n----------".format(axis))
        axis = np.array([1.0, 0.0, 0.0])

    axis = normalise(axis)
    # axis = axis/np.sqrt(np.dot(axis, axis))  # removed because normalise is nicer to read

    a = np.cos(theta/2.0)

    b, c, d = -axis*np.sin(theta/2.0)

    aa, bb, cc, dd = a*a, b*b, c*c, d*d

    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    rot_mat = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return rot_mat


def rotate(coordinates: np.array, axis: np.array, theta: float) -> np.array:
    """Takes coordinates, axis and angle and rotates the coordinates by theta around the given axes

    Args:
        coordinates: np.array, required, coordinate array to be rotated
        axis: np.array, required, axis about which to rotate
        theta: float, required, angle of rotation

    Returns:
        np.array, rotated coordinate array

    """

    rot_mat = get_rotation_matrix(axis, theta)
    new_coordinates = []
    for c in coordinates:
        new_coordinates.append(np.dot(c, rot_mat))
    return np.array(new_coordinates)


def get_angle(coordinates: np.array, radians: bool=True) -> float:
    """Takes a 3*3 numpy array of coordinates and returns an angle of coord1 -> coord2 -> coord3

    Attributes:
        coordinates: 3*3 coordinate array (required)
        radians: bool, optional, returns radians if True - degrees if False (default=True)

    Returns:
        float, angle of coordinates[0] > coordinates[1] > coordinates[2]

    """
    n_vec1 = normalise(coordinates[2, :] - coordinates[1, :])
    n_vec2 = normalise(coordinates[0, :] - coordinates[1, :])
    dp = np.dot(n_vec1, n_vec2)
    angle = np.arccos(dp)

    if radians:
        return angle
    else:
        return np.rad2deg(angle)


def normalise(vec: np.ndarray) -> np.ndarray:
    """
    Return the unit vector of vec

    Parameters
    ----------
    vec: len=3 coordinate vector to be normalised

    Returns
    -------
    unit vector of vec

    """
    m = magnitude(vec)
    if m == 0:
        log.warning("Vec = [0,0,0]")
        return vec
    return vec/m


def magnitude(v1: np.ndarray, v2: np.ndarray=np.array([0.0, 0.0, 0.0])) -> float:
    """Returns the magnitude of v1 - v2

    Args:
        v1: np.array(size=3, dtype=float), required, vector 1
        v2: np.array(size=3, dtype=float), optional, vector 2 (default = [0.0,0.0,0.0])

    Returns:
        float, magnitude of v1 - v2

    """
    vec = v1-v2
    return np.sqrt(np.sum(vec**2))


def get_all_magnitudes(coordinates: np.ndarray) -> np.ndarray:
    """Returns all the vector lengths between pairs in the coordinate array

    Uses itertools.combinations to generate pairs so the order is guaranteed to be equivalent to:

    for i in range(coordinates):
        for j in range(coordinates[i+1:])

    Args:
        coordinates: np.array, required, coordinate array

    Returns:
        np.array of all the vector lengths between coordinates in the original array

    """
    all_vecs = []
    for c1, c2 in it.combinations(coordinates, 2):
        all_vecs.append(c1-c2)

    all_magnitudes = np.sqrt((np.array(all_vecs)**2).sum(-1))
    return all_magnitudes


def get_dihedral(coordinates: np.ndarray, radians: bool=True) -> float:
    """
    Returns the dihedral angle defined by four coordinates.
    Implementation of first answer from:
    http://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates

    Returns:
        angle: float, The dihedral angle of the given set of coordinates

    Raises:
        ValueError: raised when passed a coordinate array of the wrong shape

    """
    coordinates = np.array(coordinates)
    if coordinates.shape != (4, 3):
        raise ValueError("Coordinates shape = {}. Must provide coordinates with shape = (4,3)".
                         format(coordinates.shape))

    b1 = coordinates[1] - coordinates[0]
    b2 = coordinates[2] - coordinates[1]
    b3 = coordinates[3] - coordinates[2]

    n1 = normalise(np.cross(b1, b2))
    n2 = normalise(np.cross(b2, b3))

    norm_b2 = normalise(b2)

    m1 = np.cross(n1, norm_b2)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    rads = -np.math.atan2(y, x)

    if radians:
        return rads
    else:
        degrees = np.rad2deg(rads)
        return degrees


def find_center_of_mass(coordinates: np.ndarray=None, masses: np.ndarray=None, cluster=None) -> np.ndarray:
    """Returns the center of mass of the cluster

    Args:
        coordinates: np.array, coordinate array
        masses: np.array, array of masses
        cluster: object relating to the cluster. Must have masses and coordinates attributes

    Returns:
        np.array(3), coordinates corresponding to center of mass

    """

    if cluster is not None:

        assert hasattr(cluster, "masses") and hasattr(cluster, "coordinates")

        coordinates = cluster.coordinates
        if cluster.masses is not None:
            masses = cluster.masses
        else:
            masses = get_masses(cluster.particle_names)

    com = np.zeros(3)

    for c, m in zip(coordinates, masses):
        if m == 0:
            log.warning("Particle with mass of 0 at {} not included in determining center of mass".format(c))
            continue
        wc = np.array(c) * m
        com += wc

    return com / sum(masses)
