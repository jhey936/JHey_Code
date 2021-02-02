# coding=utf-8
"""Sets up various modules written using cython and compiles them"""
from distutils.core import setup

from Cython.Build import cythonize

setup(ext_modules=cythonize("potentials/LJ_potential_c.pyx"))
