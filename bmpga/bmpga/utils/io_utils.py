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

Classes/functions to aid with file operations.

=========================================================================
"""
import os
import logging

import numpy as np

from typing import Union, List, TypeVar, Tuple

from bmpga.storage import Cluster, IdentifyFragments, Molecule
from bmpga.utils.elements import translate_symbol

# Define TypeVars for lists of clusters and lists of coords/labels to aid with typing
ClusterList = TypeVar(List[Cluster])
CoordLabels = TypeVar(List[Union[np.ndarray, List[str]]])
ListCoordLabels = TypeVar(List[CoordLabels])


class BaseWriter(object):
    """Base class writer objects should inherit from"""
    def __init__(self, log: logging.Logger=None) -> None:
        self.log = log or logging.getLogger(__name__)

    def __call__(self, structures: Union[ClusterList, ListCoordLabels],
                 filename: str) -> None:
        self.write(structures, filename)

    def write(self, structures: Union[Cluster, ClusterList, ListCoordLabels], filename: str, file_mode="a") -> None:
        """Main method called to write to files.
        Should be overridden by any subtypes

        """
        pass

    def _check_file_name(self, filename: str, file_mode: str, n_structs: int) -> None:
        """Basic error checking about the file"""
        if os.path.exists(filename):
            if file_mode == "a":
                message = "File {} already exists. {} new structures will be appended to this file."\
                    .format(filename, n_structs)
                self.log.warning(message)
            elif "w" in file_mode:
                new_fn = str(np.random.randint(0, int(1e6))) + filename
                os.rename(filename, new_fn)
                msg = "File {} exists! This file has been moved to {} rather than be overwritten with {} structures."\
                    .format(filename, new_fn, n_structs)
                self.log.warning(msg)
        else:
            msg = "Creating new file {}, and writing {} structure(s).".format(filename, n_structs)
            self.log.info(msg)

    @staticmethod
    def _get_coord_labels(clusters: ClusterList) -> ListCoordLabels:
        """Takes a list of Cluster objects and returns: List[List[coordinates, labels]]"""
        coord_labels = []
        for cluster in clusters:
            cluster.center()
            coordinates, _id, labels = cluster.get_particle_positions()
            coord_labels.append([coordinates, labels])
        return coord_labels

    def _parse_structures(self, structures: Union[Cluster, ListCoordLabels,
                                                  ClusterList]) -> Tuple[ListCoordLabels, List[float], List[int]]:
        """Type checks and parses structures into the correct format

        Attributes:
            structures: Cluster or ListCoordLabels or ClusterList, required

        Returns:
            ListCoordLabels
        """

        if isinstance(structures, Cluster):
            energies = [structures.cost]
            ids = [structures.id()]
            structures = self._get_coord_labels([structures])  # Pass List[Cluster] to _get_coord_labels
        elif isinstance(structures[0], Cluster):
            energies = [structure.cost for structure in structures]
            ids = [structure.id() for structure in structures]
            structures = self._get_coord_labels(structures)  # Pass List[Cluster] to _get_coord_labels
        elif isinstance(structures, list):
            try:
                assert isinstance(structures[0][0], np.ndarray)
            except AssertionError as error:
                message = "Must pass Cluster, List[Cluster], or List[List[np.array, labels]]\n Received: {}\n{}"\
                    .format(structures, error)
                self.log.exception(message)
                raise
            energies = ["None"] * len(structures)
            ids = ["None"] * len(structures)

        else:
            try:
                raise TypeError
            except TypeError as error:
                message = "Must pass Cluster, List[Cluster], or List[List[np.array, labels]]\n Received: {}\n{}"\
                    .format(structures, error)
                self.log.exception(message)
                raise
        return structures, energies, ids


class XYZWriter(BaseWriter):
    """Class to write XYZ format files when passes a cluster, list of clusters or List[List[coordinates, labels]]

    XYZ format files have the following format:

    <Number of atoms>
    <Comment>
    <Atom Label> <X coord> <Y coord> <Z coord>
    <Atom Label> <X coord> <Y coord> <Z coord>
    ...
    ...
    <Number of atoms>
    <Comment>
    <Atom Label> ...

    See bmpga/tests/test_data/water12.xyz for an example

    """

    def __init__(self, log: logging.Logger=None) -> None:
        super().__init__(log)

    def write(self, structures: Union[Cluster, ClusterList, ListCoordLabels],
              filename: str, file_mode: str="a") -> None:
        """Method to write XYZ format files when passes a cluster, list of clusters or List[List[coordinates, labels]]

        Args:
            structures:
            filename:
            file_mode:

        Returns:

        """

        structures, energies, ids = self._parse_structures(structures)
        self._check_file_name(filename, file_mode, len(structures))

        with open(filename, file_mode) as out_f:
            for structure, energy, _id, n in zip(structures, energies, ids, range(len(structures))):
                structure = self._format(structure)  # Formats structure into standard XYZ format
                structure[1] = "Structure {}, Energy {}, id {}\n".format(n, energy, _id)  # Inserts a comment line
                for line in structure:
                    out_f.write(line)

    @staticmethod
    def _format(raw_data: CoordLabels) -> List[str]:
        """Takes raw data in and returns a formatted list of strings in the xyz format"""
        formatted_data = [str(len(raw_data[1]))+"\n", "\n"]  # Adds the n_atoms line and blank comment line
        for label, coord in zip(raw_data[1], raw_data[0]):
            new_row = str(label) + "    " + "    ".join([str(c) for c in coord])

            formatted_data.append(new_row+"\n")  # Appends formatted data and adds newlines

        return formatted_data


class BaseReader(object):
    """Base class readers should inherit from here"""
    def __init__(self, strip_unknown: bool=False, *args, **kwargs) -> None:
        self.strip_unknown = strip_unknown

    def __call__(self, file_name: str, *args, **kwargs) -> Union[ClusterList, ListCoordLabels]:
        return self.read(file_name, *args, **kwargs)

    @staticmethod
    def _read_data(file_name: str) -> np.ndarray:
        with open(file=file_name) as open_file:
            raw_data = open_file.read()

        return np.array(raw_data.splitlines())

    def read(self, file_name: str, *args, **kwargs) -> list:
        """Main method to read the file"""
        raise NotImplementedError


class XYZReader(BaseReader):
    """Class to parse XYZ format files and return a list of cluster objects

    XYZ format files have the following format:

    <Number of atoms>
    <Comment>
    <Atom Label> <X coord> <Y coord> <Z coord>
    <Atom Label> <X coord> <Y coord> <Z coord>
    ...
    ...
    <Number of atoms>
    <Comment>
    <Atom Label> ...

    See bmpga/tests/test_data/water12.xyz for an example

    """
    def __init__(self, log=None, find_fragments=True, *args, **kwargs) -> None:

        self.log = log or logging.getLogger(__name__)


        if find_fragments:
            self.find_fragments = IdentifyFragments(*args, **kwargs)
        else:
            self.find_fragments = False

        super().__init__(*args, **kwargs)

    def read(self, file_name: str, return_clusters: bool = True,
             *args, **kwargs) -> Union[ClusterList, ListCoordLabels]:
        """Reads an XYZ format file and returns a list of Cluster objects or a list of structure coordinates
                 and labels in the format: List[List[coordinates, labels]]

        Note: does not parse the comment lines for energies. All returned clusters will have their cost set to 1.0

        Args:
            return_clusters: bool, optional, returns List[Cluster] if true or List[List[coordinates, labels]] if False
            file_name: str, required, path to file
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        Returns:
            list of Cluster objects or List[List[coordinates, labels]]

        """
        raw_data = self._read_data(file_name)
        data = self.clean(raw_data)

        number_of_particles = int(data[0][0])
        data = self.reshape(data, number_of_particles)

        self.log.info("Parsing {}, which contains {} structure(s) of {} atoms"
                      .format(file_name, data.shape[0], data.shape[1]))

        return self._parse(data, return_clusters=return_clusters)

    def _parse(self, data: Union[np.ndarray, list], return_clusters: bool = True,
               *args, **kwargs) -> Union[ClusterList, ListCoordLabels]:
        """
        Takes file data and returns cluster objects or a list of structure coordinates
                 and labels in the format: List[List[coordinates, labels]]

        Args:
            data: list, required, clean data from file
            return_clusters: bool, optional, returns List[Cluster] if true or List[List[coordinates, labels]] if False
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        Returns:
            list of Cluster objects or List[List[coordinates, labels]]

        """

        structures = []

        for structure in data:

            labels = []
            coordinates = []

            if structure[1][0:3] == ["Energy", "of", "minimum"]:
                energy = structure[1][4]
            elif structure[1][2] == "Energy":
                str_en = structure[1][3]
                energy = float(str_en.strip(","))
            else:
                energy = -1.0

            for line in structure[2:]:

                atom_label = line[0]
                atom_label = translate_symbol(atom_label)

                labels.append(atom_label)

                coordinates.append([float(c) for c in line[1:4]])

            coordinates = np.array(coordinates)
            if return_clusters:
                if self.find_fragments:
                    fragments = self.find_fragments(coordinates=coordinates, labels=labels, *args, **kwargs)
                    # Set cost to 1.0 rather than try and parse comment line
                else:
                    fragments = [Molecule(coordinates=np.array([coord]), particle_names=[label])
                                 for coord, label in zip(coordinates, labels)]

                structures.append(Cluster(molecules=fragments, cost=energy, *args, **kwargs))
            else:
                structures.append([[coordinates, labels]])

        return structures

    @staticmethod
    def clean(raw_data: np.ndarray) -> list:
        """Strips newlines and splits lines on spaces"""
        data = []
        for line in raw_data:
            line = line.strip().split(" ")
            newline = [v for v in line if v != '']
            data.append(newline)
        data = [line for line in data if line != []]
        return data

    @staticmethod
    def reshape(data: Union[np.ndarray, list], number_of_particles: int) -> np.ndarray:
        """Reshapes the data into separate structures if there are multiple structures in the same XYZ file"""
        new_shape = (int(len(data) / (number_of_particles + 2)), number_of_particles + 2)
        return np.reshape(data, newshape=new_shape)
