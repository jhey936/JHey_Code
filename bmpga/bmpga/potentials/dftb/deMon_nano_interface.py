# coding=utf-8
"""
A potential interface to the DeMon-Nano code.



=================
John Hey 12-04-19
=================
"""
import os
import shutil
import logging
import subprocess
import contextlib

import numpy as np

from unittest import TestCase

from bmpga.errors import DFTBError, OptimizationNotConvergedError
from bmpga.storage import Molecule, Cluster
from bmpga.potentials.dft.base_dft_potential import BaseDFTPotential
from bmpga.tests import test_data_path


@contextlib.contextmanager
def work_dir():
    """Context to return to the work dir on exiting a method"""
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


class DeMonDFTBPotential(BaseDFTPotential):
    """Provides an interface to the DeMon-Nano code for optimisation via DFTB"""

    def __init__(self,
                 minimize_template: str,
                 energy_template: str,
                 work_dir: str,
                 run_string: str = f"export OMP_NUM_THREADS=4; demon",  # TODO: remove this after testing
                 log: logging.Logger = logging.getLogger(__name__)) -> None:

        self.minimize_template = minimize_template
        self.energy_template = energy_template

        self.log = log

        self.particle_types = None
        self.particle_names = None
        self.natoms = None

        if work_dir[-1] != "/":
            work_dir += "/"

        super().__init__(work_dir=work_dir, run_string=run_string)

        self.last_dir = 0

    def get_directory(self, dirname="DFTB") -> str:
        """Get next directory"""
        return super().get_directory(dirname=dirname)

    # noinspection PyPep8Naming
    def run_DeMonNano(self, cluster: Cluster, dir_name: str,
                      optimize: bool = False):  # TODO move minimize and energy to here
        """Common interface to DeMonNano"""

        if dir_name is not None:
            dir_name = os.path.abspath(dir_name)
        else:
            dir_name = os.path.abspath(self.get_directory())

        inp_fname = dir_name+"/deMon.inp"
        out_fname = dir_name+"/deMon.out"

        shutil.copyfile("SCC-SLAKO", dir_name+"/SCC-SLAKO")
        shutil.copyfile("SLAKO", dir_name+"/SLAKO")

        coords, molecule_ids, atom_labels = cluster.get_particle_positions()

        Natoms = len(molecule_ids)

        xyz_formatted_coordinates = self.format_XYZ(coords, atom_labels)

        with work_dir():

            os.chdir(dir_name)  # Change into the scratch dir.

            tag_dict = {"<XYZ>": xyz_formatted_coordinates}

            if optimize:
                template = self.minimize_template
            else:
                template = self.energy_template

            self.insert_to_template(template=self.work_dir+template, out_file=inp_fname, target_dict=tag_dict)


            with open("error_file", "w") as ef:
                # self.run_string should just be the location of the deMonNano executable
                dftb_process = subprocess.Popen([self.run_string], cwd=dir_name, shell=True, stderr=ef, stdout=ef)

            exit_code = dftb_process.wait()

            # check exit code
            if exit_code != 0:
                try:
                    raise DFTBError(f"DFTB+ exited unexpectedly with exitcode: {exit_code}\n")
                except DFTBError as error:
                    self.log.exception(error)
                    raise
            else:
                self.log.debug(f"DFTB+ exited successfully. Exit code: {exit_code}")


            if optimize:

                # noinspection PyTypeChecker
                parser = DeMonNanoParser(out_fname,
                                         natoms=Natoms,
                                         logger=self.log)

                result_dict = parser.parse_DeMonNano_output()
                new_coords = result_dict["coordinates"]
                cluster.set_particle_positions((new_coords, molecule_ids, atom_labels))

            else:

                # noinspection PyTypeChecker
                parser = DeMonNanoParser(out_fname, natoms=Natoms,
                                         geometry_opt=False,
                                         logger=self.log)

                result_dict = parser.parse_DeMonNano_output()

            energy = result_dict["energy"]
            cluster.cost = energy
            # os.chdir("..")  # Come back out of the scratch dir.

            return cluster


    def get_energy(self, cluster: Cluster, *args, **kwargs) -> float:
        """Get energy at a particular coordinate"""
        raise NotImplementedError

    def minimize(self, cluster: Cluster, dir_name=None, *args, **kwargs) -> Cluster:
        """

        Args:
            cluster: bmpga.storage.Cluster, Cluster instance to be minimised
            dir_name: string, Directory to run deMon in
            *args:
            **kwargs:

        Returns:
            Cluster object with minimised geometry/energy
        """

        return self.run_DeMonNano(cluster, dir_name, optimize=True)


class DeMonNanoParser:
    """Class to parse the outputs from DeMonNano calculations"""
    def __init__(self, fname: str, geometry_opt: bool = True, natoms: int = None, logger: logging.Logger = None):
        self.fname = fname
        self.geometry_opt = geometry_opt
        self.Natoms = natoms
        self.logger = logger or logging.getLogger()


        self.convergence_line = 0
        self.data = []

    def __call__(self) -> dict:
        return self.parse_DeMonNano_output()

    def read_file(self) -> None:
        """Read the file contents and perform some basic cleaning"""

        # Read in the raw data
        with open(self.fname) as out_f:
            raw_data = out_f.readlines()

        # Clean the data and remove newlines, empty lines and preceeding/trailing whitespaces
        for line in raw_data:
            line = line.rstrip()
            line = line.lstrip()
            if line != '':
                self.data.append(line)

    def check_converged(self):
        """Check that convergence was reached, else throw OptimizationNotConvergedError """

        if self.geometry_opt:
            for line in self.data:
                if "OPTIMIZATION CONVERGED" in line:
                    self.convergence_line = self.data.index(line)
                    return

        elif not self.geometry_opt:
            for line in self.data:
                if "DFTB total energy" in line:
                    self.convergence_line = self.data.index(line)
                    return

        # If this is reached, then optimisation was not reached.
        try:
            raise OptimizationNotConvergedError(f"Optimization not converged in {os.path.curdir}!")
        except OptimizationNotConvergedError as e:
            self.logger.error(e)
            raise e

    def get_energy(self):
        """Parses DeMonNano output for the final energy"""

        if self.convergence_line is None:
            self.check_converged()

        # look for the final energy string, starting from the previously found convergence line
        for line in self.data[self.convergence_line:]:

            if "DFTB total energy" in line:
                line = line.split("=")
                energy = float(line[1])
                return energy

    def parse_structure(self):
        """Parse the DeMonNano output for the final geometry"""

        if self.convergence_line is None:
            self.check_converged()

        struct_idx = None

        # look for the final energy string, starting from the previously found convergence line
        for line in self.data[self.convergence_line:]:

            if "OPTIMIZED STRUCTURE" in line:
                self.get_natoms()
                struct_idx = self.data.index(line)+2
                break
            else:
                continue

        if struct_idx is None:
            try:
                raise DFTBError("Structure Not Optimised")
            except DFTBError as error:
                self.logger.exception(error)
                raise error

        if not self.Natoms:
            self.get_natoms()

        labels, coords = [], []


        for line in self.data[struct_idx:struct_idx+self.Natoms]:
            line = line.split(" ")
            line = [s for s in line if s != ""]
            labels.append(line[1])
            coords.append([float(c) for c in line[3:6]])

        return labels, coords

    def get_natoms(self):
        """Get the number of atoms"""

        for line in self.data:
            if "Number of DFTB atoms" in line:
                line = line.split(":")
                self.Natoms = int(line[1])

    # noinspection PyPep8Naming
    def parse_DeMonNano_output(self) -> dict:
        """Method to parse DeMonNano outputs for structure and energy"""
        self.read_file()
        self.check_converged()

        output_dict = dict()

        output_dict["energy"] = self.get_energy()

        # Get the geometry from the output if required
        if self.geometry_opt:
            labels, coords = self.parse_structure()
            output_dict["labels"] = labels
            output_dict["coordinates"] = coords

        return output_dict


"""Will skip some tests with messages if deMon executable not found in $PATH"""


class TestDeMonNanoPotential(TestCase):
    """Probably not good practice, but I am treating the DeMonNano parser as part of the potential for testing..."""
    @classmethod
    def setUpClass(cls) -> None:
        """Checks for the DeMonNano executable"""
        cls.executable = shutil.which("deMon")


    def setUp(self) -> None:
        """Setup the path to the test_data"""
        self.test_data_path = test_data_path

    def test_parser_good_opt(self):

        # noinspection PyArgumentEqualDefault
        parser = DeMonNanoParser(self.test_data_path+"/deMon_good.out", geometry_opt=True)
        output_dict = parser()

        self.assertTrue(parser.Natoms == 6)
        self.assertEqual(output_dict["energy"], -12.44519867)
        self.assertListEqual(output_dict["labels"], ['C', 'O', 'O', 'O', 'H', 'H'])
        self.assertListEqual(output_dict["coordinates"],
                             [[-0.253969, 0.011832, 0.025035],
                              [0.603696, -0.708116, 0.383405],
                              [-1.110592, 0.725448, -0.330598],
                              [3.29822, 0.433619, 0.087403],
                              [2.487505, -0.079397, 0.24634],
                              [3.976553, -0.184672, 0.396278]])

    def test_parser_good_no_opt(self):

        parser = DeMonNanoParser(self.test_data_path + "/deMon_good.out", geometry_opt=False)
        output_dict = parser()

        self.assertIsNone(parser.Natoms)  # check that a structure has not been read in
        self.assertEqual(output_dict["energy"], -12.44519867)

    def test_parser_not_converged(self):

        with self.assertRaises(OptimizationNotConvergedError):
            parser = DeMonNanoParser(self.test_data_path + "/deMon_not_opt.out", geometry_opt=True)
            parser()

    def test_parser_no_struct(self):


        parser = DeMonNanoParser(self.test_data_path + "/deMon_energy_only.out", geometry_opt=False)
        output_dict = parser()


        self.assertIsNone(parser.Natoms)  # check that a structure has not been read in
        self.assertEqual(output_dict["energy"], -12.29809526)

        # self.skipTest("Parser test not implemented")

    def test_DeMonNano_exited_unsucessfully(self):
        self.skipTest("Unsucessful exited test not implemented")

    def test_minimise(self):
        if self.executable is None:
            self.skipTest("Executable not found. Minimise test skipped.")

        testdir = "test_dftb"
        if os.path.exists(testdir):
            shutil.rmtree(testdir)
        os.mkdir(testdir)

        clus1 = Cluster(molecules=[Molecule(particle_names=["H", "H", "O"],
                                            coordinates=np.array([[0.0, 0.0, 1.0],
                                                                  [0.0, 0.0, -1.0],
                                                                  [0.0, 0.0, 0.0]])),
                                   Molecule(particle_names=["H", "H", "O"],
                                            coordinates=np.array([[2.0, 0.0, 1.0],
                                                                  [2.0, 0.0, -1.0],
                                                                  [2.0, 0.0, 0.0]]))
                                   ]
                        )

        pot = DeMonDFTBPotential(minimize_template=self.test_data_path+"/dftb_in.hsd",
                                 energy_template="NONE", work_dir=".",
                                 run_string=self.executable)

        clus2 = pot.minimize(clus1, dir_name=testdir)

        coords, mol_ids, atom_names = clus2.get_particle_positions()

        self.assertListEqual(atom_names, ["H", "H", "O", "H", "H", "O"])

        if os.path.exists(testdir):
            shutil.rmtree(testdir)


#     def test_parser_good(self):
#
#         energy, coords = DFTBPotential.parse_dftbp_output(self.test_data_path + "/dftb_good_output.test",
#                                                           out_coords_fname=self.test_data_path+"/dftb_good_geom.xyz")
#
#         self.assertListEqual(list(coords[-1]), [1.150167,   -0.16129964,  0.13635255])
#
#         self.assertEqual(energy, -2448.9224)
#
#     def test_parser_no_struct(self):
#
#         with self.assertRaises(IOError):
#             DFTBPotential.parse_dftbp_output(self.test_data_path + "/dftb_good_output.test",
#                                              out_coords_fname=self.test_data_path+"/dftb_bad_geom.xyz")
#
#     def test_parser_unminimised_output(self):
#
#         with self.assertRaises(DFTBError):
#             DFTBPotential.parse_dftbp_output(self.test_data_path + "/dftb_bad_output.test",
#                                              out_coords_fname=self.test_data_path+"/dftb_good_geom.xyz")
#
#     def test_dftb_exited_unsuccessfully(self):
#
#         raise self.skipTest("Test not implemented")
#
#     def test_minimise(self):
#

