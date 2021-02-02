# coding=utf-8
"""

bmpga: A program for finding global minima
Copyright (C) 2018- ; John Hey
This file is part of bmpga.

bmpga is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License V3.0 as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

bmpga is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License V3.0 for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


=========================================================================
John Hey (Created: 5-11-18)

Takes a Molecule object relating to a peptide and performs various analysis

=========================================================================
"""
import logging

import numpy as np
import networkx as nx
import itertools as it

from unittest import TestCase
from typing import Tuple

from bmpga.storage.molecule import Molecule, FlexibleMolecule
from bmpga.utils.geometry import get_dihedral


class Peptide(FlexibleMolecule):
    """
    # todo add docstring
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger()

    def __call__(self, molecule: Molecule, *args, **kwargs) -> dict:
        return self.calculate_dihedrals()

    def find_ends(self) -> Tuple:
        """Traverses the graph and identifies the likely C and N atoms for the terminals of the backbone.

        returns:
            Tuple of lists containing likely terminal atoms to be checked.

        """

        carboxylates = []
        amines = []
        # noinspection PyUnboundLocalVariable
        for atom in self.graph.nodes():

            '''Look through the graph for carboxylate and amino groups and returns '''

            if atom.label == "C":   # todo: look for capped peptides
                neighbours = self.graph.neighbors(atom)

                # noinspection PyPep8Naming
                Os = [a for a in neighbours if a.label == "O" or a.label == "OXT"]

                for O in Os:
                    neighbours = list(self.graph.neighbors(O))

                    if len(neighbours) > 1:
                        labels = [a.label for a in neighbours]
                        if "H" not in labels or len(labels) > 2:
                            Os.remove(0)

                if len(Os) == 2:
                    carboxylates.append(Os[0])

            if atom.label == "N":
                neighbours = self.graph.neighbors(atom)

                # noinspection PyPep8Naming
                Hs = [a for a in neighbours if a.label == "H"]
                if len(Hs) == 3:
                    amines.append(Hs[0])
        return amines, carboxylates

    def find_backbone(self):

        amines, carboxylates = self.find_ends()

        '''Identifies the peptide backbone'''

        peptide_backbone = False
        for possible_N_term, possible_C_term in it.product(amines, carboxylates):

            path = nx.shortest_path(self.graph, possible_C_term, possible_N_term)

            path_labels = [a.label for a in path]

            peptide_backbone = self.check_peptide(path)

            if peptide_backbone:
                peptide_backbone = path
                break

        # noinspection PyUnboundLocalVariable
        if not peptide_backbone:
            class PeptideNotFound(Exception):
                pass
            try:
                raise PeptideNotFound(f"{[a.label for a in self.graph]}")
            except PeptideNotFound as e:
                self.log.exception(PeptideNotFound(e))

        return peptide_backbone

    @staticmethod
    def check_peptide(path) -> bool:

        path = [a.label for a in path]
        peptide_list = ["C", "C", "N"]

        peptide_bonds = 0

        i = 0

        found_peptides = False
        while i <= len(path):

            test_path = path[i:i+3]
            # print(test_path)
            if test_path == peptide_list:

                found_peptides = True
                # print(found_peptides)
                peptide_bonds += 1
                i += 3

            elif found_peptides:
                # print(3*peptide_bonds, len(path)-2, 3*peptide_bonds == len(path)-2)
                if 3*peptide_bonds == len(path)-2:
                    # print(f"{path}, Returning True")
                    return True
                else:
                    break
            else:
                i += 1

        return False

    # noinspection PyMethodOverriding
    def calculate_dihedrals(self, path=None) -> dict:
        """Returns all phi, psi angles along the dipeptide backbone.

        :param path:
        :return:
        """
        if path is None:
            path = self.find_backbone()

        phi_labels = ["C", "C", "N", "C"]
        psi_labels = ["N", "C", "C", "N"]
        omega_labels = ["C", "N", "C", "C"]
        i = 0

        omega_angle, psi_angle, phi_angle = [], [], []
        while i <= len(path):

            i += 1
            test_path = path[i:i+4]
            test_path_labels = [a.label for a in test_path]
            # print(test_path)
            coords = np.array([a.coordinate for a in test_path])

            if test_path_labels == psi_labels:
                psi_angle.append(np.degrees(get_dihedral(coordinates=coords)))
            elif test_path_labels == phi_labels:
                phi_angle.append(np.degrees(get_dihedral(coordinates=coords)))
            elif test_path_labels == omega_labels:
                omega_angle.append(np.degrees(get_dihedral(coordinates=coords)))

        return {"phi": phi_angle, "psi": psi_angle, "w": omega_angle}


class TestAnalyseDihedrals(TestCase):

    @classmethod
    def setUpClass(cls):
        """Creates a test Molecule"""
        from bmpga.utils.io_utils import XYZReader
        from bmpga.tests import test_data_path

        reader = XYZReader()

        mol = reader.read(test_data_path+"/NLYS_CASP_TEST.xyz")[0].molecules[0]
        cls.good_peptide_1 = Peptide(
            coordinates=mol.coordinates,
            particle_names=mol.particle_names
        )

        mol = reader.read(test_data_path + "/bad_peptide.xyz")[0].molecules[0]
        cls.bad_peptide_ends = Peptide(
            coordinates=mol.coordinates,
            particle_names=mol.particle_names
        )
        mol = reader.read(test_data_path + "/bad_peptide.xyz")[1].molecules[0]
        cls.bad_peptide_chain = Peptide(
            coordinates=mol.coordinates,
            particle_names=mol.particle_names
        )

    # noinspection PyPep8Naming
    def test_find_ends_good(self):
        N_terms = [a.label for a in self.good_peptide_1.find_ends()[0]]
        C_terms = [a.label for a in self.good_peptide_1.find_ends()[1]]

        self.assertTrue(N_terms == ["H", "H"])
        self.assertTrue(C_terms == ["O", "O"])

    def test_find_ends_bad_ends(self):
        # this one only returns one amide end
        self.assertEqual(len(self.bad_peptide_ends.find_ends()[0]), 1)

    def test_find_chain_good(self):
        self.fail("Test not implemented")


if __name__ == "__main__":
    from bmpga.utils.io_utils import XYZReader
    from os.path import expanduser

    reader = XYZReader()
    cluster = reader.read(expanduser("~/Software/bmpga/bmpga/tests/test_data/NLYS_CASP_TEST.xyz"))[0]
    print(cluster.molecules)
    peptide = Peptide(coordinates=cluster.molecules[0].coordinates,
                      particle_names=cluster.molecules[0].particle_names)

    dihedrals = list(peptide.calculate_dihedrals().values())
    print(np.degrees(dihedrals))

