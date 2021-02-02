# coding=utf-8
"""Class user can use to define their system.

The GA will then use this class to generate clusters to initialise the pool.
"""  # TODO: Ensure the DefineSystem class is well documented
import logging

from typing import List, TypeVar

from bmpga.storage import Cluster, Molecule
from bmpga.mutation import RandomCluster
from bmpga.utils.io_utils import XYZReader

ListMol = TypeVar(List[Molecule])


class DefineSystem(object):
    """
    Main method for defining the system to be optimised.
    """
    def __init__(self, numbers_of_molecules: List[int],
                 molecules: ListMol=None,
                 template_files: str or List[str]=None,
                 box_length: float=None, fixed_box: bool=False,
                 log: logging.Logger=None, *args, **kwargs):

        self.numbers_of_molecules = numbers_of_molecules

        self.log = log or logging.getLogger(__name__)
        self.RandomCluster = RandomCluster(log=self.log, box_length=box_length, fixed_box=fixed_box)

        self.all_molecules = []

        try:
            assert isinstance(molecules, List) or isinstance(template_files, List) or isinstance(template_files, str)
        except AssertionError as error:
            self.log.exception("Must pass either coordinate file(s) or a list of Molecules\n{}".format(error))
            raise

        if isinstance(molecules, List):
            self.initial_molecules = molecules  # Users can pass List[Molecules]
        else:
            self.initial_molecules = []

        if template_files is not None:
            self.get_templated_molecules(template_files, *args, **kwargs)

        self._get_all_molecules()
        self.base_cluster = Cluster(cost=0.0, molecules=self.all_molecules, step=0, minimum=False)

    def __call__(self) -> Cluster:
        return self.get_random_cluster()

    def _get_all_molecules(self) -> None:
        """Takes list of Molecules and the number of each that is required, and makes the final list of molecules"""
        try:
            assert len(self.numbers_of_molecules) == len(self.initial_molecules)
        except AssertionError as error:
            self.log.exception(
                "Must pass an equal number of molecular templates and their numbers! {0} != {1}\n{2}"
                    .format(len(self.initial_molecules), len(self.all_molecules), error))
            raise

        for mol, number in zip(self.initial_molecules, self.numbers_of_molecules):
            self.all_molecules.extend([mol]*number)

    def get_templated_molecules(self, file_names, *args, **kwargs) -> None:
        """Takes a file name or list of file names and parses them into initial_molecules which
               then extend self.initial_molecules

        Guesses the format based on the file extension

        Currently only supports .XYZ format files

        Plan to extend to .pdb and .mol2 as well as some of the DFT outputs we will parse elsewhere.

        Args:
            file_names: str or List[str], required, filename or list of file_names to be parsed

        Returns:
            None

        Raises:
            NotImplementedError if a .pdb or .mol2 file is passed in.

        """

        # Bit of a hack to allow single file_names
        if isinstance(file_names, str):
            file_names = [file_names]

        molecules = []
        for f_name in file_names:
            extension = f_name.split(".")[1].lower()

            structures = []

            if extension == "xyz":
                reader = XYZReader(*args, **kwargs)
                structures = reader(f_name, return_clusters=False)

            elif extension == "pdb":
                try:
                    raise NotImplementedError("pdb files are not yet supported")
                except NotImplementedError as error:
                    self.log.exception(error)
                    raise

            elif extension == "mol2":
                try:
                    raise NotImplementedError("mol2 files are not yet supported")
                except NotImplementedError as error:
                    self.log.exception(error)
                    raise

            self.log.info("""{} structures found in {}. Note that all structures will be used."""
                          .format(len(structures), f_name))

            if not isinstance(structures, list):
                structures = [structures]

            for mol in structures:
                molecules.append(Molecule(coordinates=mol[0][0], particle_names=mol[0][1]))

        self.log.info("After parsing the file(s): {}.\nThese initial_molecules are being added to the system: {}"
                      .format(file_names, molecules))
        self.initial_molecules.extend(molecules)

    def get_random_cluster(self) -> Cluster:
        """Invokes calls to bmpga.mutate.RandomCluster to generate randomised clusters of non-overlapping molecules

        Returns:
            Cluster, guaranteed new and unique random cluster (new and unique ids etc)

        """
        return self.RandomCluster(self.base_cluster)
