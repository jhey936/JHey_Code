# coding=utf-8
"""
Provides some utilities to make life easier for us chemists.
"""
import numpy as np

from bmpga.utils.elements import translate_symbol


# noinspection PyPep8
def get_masses(atom_names: list) -> np.ndarray:
    """Returns a list of atomic masses from a list of atom names

    Masses taken from 'https://gist.github.com/gtamazian/7924945' on 12/05/18

    Args:
         atom_names: list(str), required, list of atomic names as strings

    Returns:
         list(float) of atomic masses. Order is preserved.

    Raises:
         KeyError if atom name is not recognised.
    """
    # TODO: Consider these instead http://www.ccp4.ac.uk/dist//checkout/arcimboldo/src/geometry.py
    # TODO: Rewrite this to read in data from a csv file.
    atomic_masses = dict(H=1.01, He=4.00, Li=6.94, Be=9.01, B=10.81, C=12.01,
                         N=14.01, O=16.00, F=19.00, Ne=20.18, Na=22.99, Mg=24.31,
                         Al=26.98, Si=28.09, P=30.97, S=32.07, Cl=35.45, Ar=39.95,
                         K=39.10, Ca=40.08, Sc=44.96, Ti=47.87, V=50.94, Cr=52.00,
                         Mn=54.94, Fe=55.85, Co=58.93, Ni=58.69, Cu=63.55, Zn=65.39,
                         Ga=69.72, Ge=72.61, As=74.92, Se=78.96, Br=79.90, Kr=83.80,
                         Rb=85.47, Sr=87.62, Y=88.91, Zr=91.22, Nb=92.91, Mo=95.94,
                         Tc=98.00, Ru=101.07, Rh=102.91, Pd=106.42, Ag=107.87,
                         Cd=112.41, In=114.82, Sn=118.71, Sb=121.76, Te=127.60,
                         I=126.90, Xe=131.29, Cs=132.91, Ba=137.33, La=138.91,
                         Ce=140.12, Pr=140.91, Nd=144.24, Pm=145.00, Sm=150.36,
                         Eu=151.96, Gd=157.25, Tb=158.93, Dy=162.50, Ho=164.93,
                         Er=167.26, Tm=168.93, Yb=173.04, Lu=174.97, Hf=178.49,
                         Ta=180.95, W=183.84, Re=186.21, Os=190.23, Ir=192.22,
                         Pt=195.08, Au=196.97, Hg=200.59, Tl=204.38, Pb=207.2,
                         Bi=208.98, Po=209.00, At=210.00, Rn=222.00, Fr=223.00,
                         Ra=226.00, Ac=227.00, Th=232.04, Pa=231.04, U=238.03,
                         Np=237.00, Pu=244.00, Am=243.00, Cm=247.00, Bk=247.00,
                         Cf=251.00, Es=252.00, Fm=257.00, Md=258.00, No=259.00,
                         Lr=262.00, Rf=261.00, Db=262.00, Sg=266.00, Bh=264.00,
                         Hs=269.00, Mt=268.00,
                         LJ=1.00, DUMMY=0.0, EP=0.0, D=2.01410178, EPW=0.0)  # mass for D taken from wikipedia
    atom_names = [translate_symbol(a) for a in atom_names]

    if atom_names is None:
        raise TypeError("No atom names supplied to get_masses!")
    elif type(atom_names) is not list:
        atom_names = [atom_names]

    masses = []
    for name in atom_names:
        try:
            masses.append(atomic_masses[name])
        except KeyError:
            raise KeyError("Atom {} not recognised".format(name))
    return np.array(masses)


def get_mass(molecule) -> float:
    """Returns the sum of masses of a fragment"""
    try:
        return sum(molecule.masses)
    except TypeError:
        try:
            return sum(get_masses(molecule.particle_names))
        except TypeError:
            raise TypeError("No labels or masses attached to: {}".format(molecule))


