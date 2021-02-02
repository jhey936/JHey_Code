# coding=utf-8
"""
Defines some water models

"""
import json
import copy
import unittest

import numpy as np

from typing import Tuple

from bmpga.storage.molecule import Molecule
import bmpga.potentials.parameters as parameters


class GenerateOPLSMolecule:
    """
    Generates molecules and OPLS paramters for pre-parameterised molecules.
    Look in bmpga.potentials.OPLS_parameters for the parameterised species.
    """
    def __init__(self, name):
        self.atoms, self.coordinates, self.charges, self.sigma, self.epsilon = self.read_json(name=name)
        self.molecule = Molecule(coordinates=copy.deepcopy(self.coordinates),
                                 particle_names=copy.deepcopy(self.atoms))

    @staticmethod
    def read_json(name: str):
        """Reads in the """
        param_loc = parameters.__file__[:-11]
        with open(f"{param_loc}{name}.json") as json_file:
            data = json.load(json_file)
        return data["atoms"], data["coordinates"], data["charge"], data["sigma"], data["epsilon"]

    def __call__(self, *args, **kwargs):
        return self.get_mol_params()

    def get_mol_params(self) -> Tuple[Molecule, np.ndarray, np.ndarray, np.ndarray]:
        """

        Returns:
            molecule, charges, sigma, epsilon
        """
        new_molecule = copy.deepcopy(self.molecule)
        new_molecule.translate(np.random.uniform(1.0, 10.0, 3))
        new_molecule.rotate(np.random.uniform(0.2, 10.0, 3), np.random.random()*2*np.pi)

        return new_molecule, copy.deepcopy(self.charges), copy.deepcopy(self.sigma), copy.deepcopy(self.epsilon)


class TestGenerateOPLSPotential(unittest.TestCase):

    def test_tip4p(self):

        get_tip4p = GenerateOPLSMolecule("tip4p")

        self.assertListEqual(get_tip4p.atoms, ["O", "H", "H", "EP"])

    def test_tip4p_call(self):
        get_tip4p = GenerateOPLSMolecule("tip4p")
        self.assertIsInstance(get_tip4p()[0], Molecule)

