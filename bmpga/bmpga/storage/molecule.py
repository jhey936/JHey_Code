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
John Hey (Created: 2018)

Defines Fragments and Particles and provides a class to identify fragments within a cluster

=========================================================================
"""

import copy
import itertools
import uuid
import logging

import numpy as np

import networkx as nx
import itertools as it

from typing import List, Union, Tuple

from bmpga.utils.chem import get_masses
from bmpga.errors import CantIdentifyHeadTail
from bmpga.utils.elements import ParticleRadii
from bmpga.utils.geometry import get_all_magnitudes, rotate, find_center_of_mass, get_dihedral, magnitude


class Particle(object):
    """Particle class"""
    def __init__(self, label: str=None, coordinate: np.array=None) -> None:
        """

        Args:
            label: str, required, particle label
            coordinate: coordinate
        """

        self.label = label
        self.coordinate = coordinate

    def __hash__(self) -> int:
        return int("".join([str(crd).replace(".", "0").replace("-", "0").replace("e", "0") for crd in self.coordinate]))

    def __cmp__(self, other) -> bool:
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return f"{self.label}@{self.coordinate}"


class Molecule(object):
    """
    Defines the basic molecule object.

    """
    def __init__(self,
                 coordinates: np.ndarray,
                 particle_names: List[str]=None,
                 masses: Union[List, np.ndarray]=None) -> None:

        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.particle_names = particle_names or ["LJ"]*len(coordinates)

        if masses is not None:
            self.masses = np.array(masses, dtype=np.float64)
        else:
            self.masses = None

        self._id = uuid.uuid4().int  # Gives unique id

        self.check_valid()

        if (masses is None) and (particle_names is not None):
            self.masses = self.get_masses()

        if self.masses is not None:
            self.mass = np.sum(self.masses)

        self.name = self.get_name(particle_names)

    def to_graph(self) -> nx.Graph:  # todo Test molecule.to_graph()!
        """

        Returns:
            networkx.Graph object from IdentifyFragments()

        """
        identify = IdentifyFragments()

        return identify.find_fragments(self.coordinates, self.particle_names, graph=True)

    def get_masses(self) -> np.ndarray:
        """Wraps get_masses()"""
        return get_masses(self.particle_names)

    def __repr__(self) -> str:
        return "Molecule {}: {}".format(self._id, "".join(self.get_name(self.particle_names)))

    def check_valid(self) -> None:
        """Basic checks on the validity of the fragment

        Returns:
            None

        Raises:
            AttributeError, raised if checks are failed

        """
        if (self.particle_names is not None) and (len(self.particle_names) != len(self.coordinates)):
            # self.log.critical()
            raise AttributeError("Coordinates and atomNames are of different lengths. {} & {}".
                                 format(len(self.coordinates), len(self.particle_names)))

    def __eq__(self, other) -> bool:
        # IDE was complaining that I was using a private attribute by accessing other._id
        return self._id == other.__hash__()

    def __hash__(self) -> int:
        return self._id

    def get_name(self, particle_names: List[str]) -> str:
        """Returns a name for the molecule. Should resolve to the same name for the same initial_molecules"""

        if particle_names is not None:
            return "".join(sorted(particle_names))
        elif self.masses is not None:
            return "".join(sorted([str(M) for M in self.masses]))
        else:
            return "Unknown"

    def rotate(self, axis: np.ndarray, theta: float) -> None:
        """Translates self.center_of_mass to origin, applies rotation then translates self back

        Args:
            axis: np.array(3), required, axis about which to rotate
            theta: float, required, angle of rotation

        Returns:
            None

        """
        center_of_mass = self.get_center_of_mass()
        self.translate(-center_of_mass)

        self.coordinates = rotate(self.coordinates, axis, theta)

        self.translate(center_of_mass)

    def translate(self, vec: np.ndarray) -> None:
        """Translates the fragment by vec

        Args:
            vec: np.array(3), required, vector by which to translate

        Returns:
            None
        """
        self.coordinates += vec

    def center(self) -> None:
        """Moves the center of mass of the molecule to the origin

        Returns:
            None
        """
        self.translate(-self.get_center_of_mass())

    def get_center_of_mass(self) -> np.ndarray:
        """Returns the center of mass of the cluster

        Returns:
            np.array(3), coordinates of the center of mass

        """

        if self.masses is None:
            try:
                self.masses = get_masses(self.particle_names)
            except KeyError:  # TODO: test the error checking in get_center_of_mass
                raise

        return find_center_of_mass(coordinates=self.coordinates, masses=self.masses)

    def id(self) -> int:
        """Returns the integer id of the molecule"""
        return self._id


class IdentifyFragments(object):
    """Class to identify initial_molecules/fragments within clusters"""
    def __init__(self, error: float = 1.03,
                 radii: ParticleRadii = None,
                 molecule_list=None,
                 *args, **kwargs) -> None:

        self.log = logging.getLogger(__name__)

        # Is 3% a good default error margin? Too much?
        assert type(error) is float
        self.error = error  # Amount to fudge atomic radii by for calculating cutoffs (default 3%)?

        if radii is not None:
            self.radii = radii
            assert isinstance(self.radii, ParticleRadii)
        else:
            self.radii = ParticleRadii(*args, **kwargs)

        if molecule_list is not None:
            self.molecule_name_list = [mol.name for mol in molecule_list]
        else:
            self.molecule_name_list = None

        self.cutoffs = {}  # We will cache lookups as we make them for speed later on

    def get_cutoff_distance(self, species1: str, species2: str) -> float:
        """Returns the cached cutoff distance computed for identifying a bond between species 1 and 2

        r_cut = (radius(species1) + radius(species2)) * self.error

        (Default self.error is 1.05)

        Args:
            species1: str, required, symbol of the first species
            species2: str, required, symbol of the second species

        Returns:
            r_cut: float, computed cutoff distance

        """
        try:
            return self.cutoffs[species1 + species2]

        except KeyError:
            radius1 = self.radii(species1)
            radius2 = self.radii(species2)
            cutoff = (radius1+radius2)*self.error  # Sum of (hopefully) covalent radii + 5%
            self.cutoffs[species1 + species2] = cutoff  # Cache result for next time
            self.cutoffs[species2 + species1] = cutoff  # Cache the reverse result for next time
            return self.cutoffs[species1 + species2]  # Return the new cutoff

    def find_fragments(self, coordinates: np.ndarray, labels: list,
                       graph=False, **kwargs) -> Union[list, nx.OrderedGraph]:
        """Reduces a set of labeled coordinates to a graph with edges drawn between species which are closer than a
               distance cutoff (calculated from covalent radii) then extracts connected fragments from that graph.

        Args:
            coordinates: np.array, required, cartesian coordinate array
            labels: list[str], required, list of particle labels
            graph: bool, (optional),
            **kwargs: additional keyword arguments

        Returns:
            list of Molecule objects or an nx.OrderedGraph object

        """

        if len(coordinates) != len(labels):  # TODO: test the assertion in find_fragments
            raise AssertionError("Coordinates and labels of different lengths! {} & {}".
                                 format(coordinates.shape, len(labels)))

        fragment_graph = nx.OrderedGraph()

        for label, coordinate in zip(labels, coordinates):
            graph_node = Particle(label, coordinate)
            fragment_graph.add_node(graph_node)

        all_r = get_all_magnitudes(coordinates)

        for (particle1, particle2), r in zip(it.combinations(fragment_graph.nodes, 2), all_r):

            if r <= self.get_cutoff_distance(particle1.label, particle2.label):
                fragment_graph.add_edge(particle1, particle2, weight=r, **kwargs)

        if graph:
            return fragment_graph
        else:
            return self.graph_to_fragments(fragment_graph)

    def graph_to_fragments(self, graph) -> list:
        """Extracts molecules from the graph

        Args:
            graph: nx.OrderedGraph

        Returns:
            list[Fragments]

        """

        molecules = []
        for subgraph in nx.connected_components(graph):
            molecle = self.subgraph_to_fragment(subgraph)

            if self.molecule_name_list is not None:
                if molecle.name not in self.molecule_name_list:
                    self.log.info(f"{molecle.name} not in {self.molecule_name_list}! Attempting to fix")


            molecules.append(molecle)
        return molecules

    @staticmethod
    def subgraph_to_fragment(subgraph) -> Molecule:
        """Turns a subgraph into a Fragment

        Args:
            subgraph: set of Particle objects comprising the subgraph

        Returns:
            Molecule

        """
        return Molecule(np.array([p.coordinate for p in subgraph]), [p.label for p in subgraph])

    def __call__(self, coordinates, labels, *args, **kwargs) -> list:
        return self.find_fragments(coordinates, labels, **kwargs)


class FlexibleMolecule(Molecule):
    """
    Class to describe flexible molecules and provide an interface to allow modification of internal degrees of freedom.

    """
    def __init__(self, head_atoms: Tuple=None, backbone: List=None, *args, **kwargs) -> None:

        super().__init__(* args, **kwargs)
        self.log = logging.getLogger()

        self.graph = self.to_graph()
        self.head_atoms = head_atoms or self.identify_head_tail()
        self.backbone = backbone or self.identify_chain()

    def find_backbone(self) -> List:
        """
        Identifies the atoms which comprise the backbone of the molecule

        Args: #todo write docstring


        Returns:

        """

        if self.head_atoms is None:
            self.head_atoms = self.identify_head_tail()

        self.backbone = nx.shortest_path(self.graph, source=self.head_atoms[0], target=self.head_atoms[1])
        return self.backbone

    def identify_head_tail(self) -> Tuple:
        """
        Traverses the molecular graph and identifies the probable head and tail.
        Identifies head and tail based on the pair of nodes with the greatest path distance.

        Args:


        Returns:
            Tuple(head atom, tail atom)
        """

        eccentricity = nx.eccentricity(self.graph)

        ends = []
        for node in eccentricity.keys():

            if eccentricity[node] == max(eccentricity.values()):
                ends.append(node)

        if len(ends) > 2:

            max_dist = 0.0
            potential_ends = ()
            for pair in itertools.combinations(ends, 2):
                dist = magnitude(pair[0].coordinate, pair[1].coordinate)

                if dist > max_dist:
                    max_dist = copy.deepcopy(dist)
                    potential_ends = pair
                else:
                    continue
            ends = potential_ends

            if len(ends) == 2:
                self.head_atoms = ends
                return tuple(ends)
            else:
                raise CantIdentifyHeadTail(
                    f"""identify_head_tail can't determine head and tail atoms automatically: 
                    too many atoms! {[str(a) for a in ends]}\n""")

        elif len(ends) < 2:
            raise CantIdentifyHeadTail(
                f"""identify_head_tail can't determine head and tail atoms automatically: "
                too few atoms! {[str(a) for a in ends]}\n"""
            )
        elif len(ends) == 2:
            self.head_atoms = ends
            return tuple(ends)

    def identify_chain(self) -> List:
        """

        Returns:

            List of nodes that compise the chain

        """
        if self.head_atoms is None:
            self.head_atoms = self.identify_head_tail()

        return nx.shortest_path(self.graph, source=self.head_atoms[0], target=self.head_atoms[1])

    def calculate_dihedrals(self):
        """
        Returns:
            List of dihdral angles from the backbone

        """

        if self.backbone is None:
            self.backbone = self.identify_chain()

        dihedrals = []

        for i, _ in enumerate(self.backbone):
            if i+4 <= len(self.backbone):
                atoms = self.backbone[i: i+4]
                coordinates = np.array([atom.coordinate for atom in atoms])
                dihedrals.append(get_dihedral(coordinates=coordinates, radians=False))
            else:
                return dihedrals


def copy_molecules(molecules: List[Molecule]) -> List[Molecule]:
    """Returns a list of new Molecule objects copied from the input. (Same labels and coordinates, new ids)"""
    new_molecules = []
    for m in molecules:
        new_molecules.append(Molecule(coordinates=copy.deepcopy(m.coordinates),
                                      particle_names=copy.deepcopy(m.particle_names),
                                      masses=copy.deepcopy(m.masses)))
    return new_molecules
