# coding=utf-8
"""
A potential interface to the DFTB+ code.



=================
John Hey 12-04-19
=================
"""
import os
import shutil
import logging
import subprocess

import numpy as np

from shutil import copyfile
from unittest import TestCase

from bmpga.errors import DFTBError
from bmpga.storage.cluster import Cluster
from bmpga.storage.cluster import Molecule
from bmpga.utils.io_utils import XYZReader
from bmpga.potentials.dft.base_dft_potential import BaseDFTPotential


class DFTBPotential(BaseDFTPotential):
    """Provides an interface to the DFTB+ code for optimisation via DFTB"""

    def __init__(self,
                 minimize_template: str,
                 energy_template: str,
                 work_dir: str,
                 run_string: str = f"export OMP_NUM_THREADS=20; dftb+",  # TODO: remove this after testing
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
        return super().get_directory(dirname)

    @staticmethod
    def write_gen(outfile, struct_dict) -> None:
        """Write gen format structure file"""
        with open(outfile, "w") as of:
            of.write(f"{struct_dict['Natoms']} C \n")
            of.write(f"{' '.join(struct_dict['names'])} \n")

            for idx, (atom_type, coord) in enumerate(zip(struct_dict["types"], struct_dict["coords"])):
                of.write("   ".join([str(idx)] + [str(atom_type + 1)] + [str(i) for i in coord]) + "\n")

    @staticmethod
    def parse_dftbp_output(filename, out_coords_fname, hartree=False):
        """Parse dftb+ output"""
        with open(filename) as outf:
            data = outf.readlines()

        converged = False

        energies = []
        for line in data:
            if "Geometry converged" in line:
                converged = True
            if "Total Energy:" in line:
                line = [a for a in line.strip().split(" ") if a != ""]
                energies.append(line)

        if not converged:
            raise DFTBError(f"DFTB calculation did not converge:\n{filename}")

        if hartree:
            energy = energies[-1][2]
        else:
            energy = energies[-1][4]

        energy = float(energy)

        reader = XYZReader()
        coords = reader.read(out_coords_fname, return_clusters=False)[-1][0][0]

        coords = np.array(coords, dtype=np.float64)

        # print(coords, len(coords))
        return energy, coords


    def minimize(self, cluster: Cluster, dir_name=None, *args, **kwargs) -> Cluster:
        """
        
        Args:
            cluster: 
            dir_name: 
            *args: 
            **kwargs: 

        Returns:

        """  # TODO finish docstring
        dir_name = dir_name or os.path.abspath(self.get_directory())

        copyfile(self.minimize_template, dir_name+"/dftb_in.hsd")

        coords, molecule_ids, atom_labels = cluster.get_particle_positions()

        if self.particle_types is None:
            self.particle_types = []
            self.particle_names = []

            for l in atom_labels:
                if l not in self.particle_names:
                    self.particle_names.append(l)
                self.particle_types.append(self.particle_names.index(l))
            self.natoms = len(self.particle_types)

        inp_gen_fname = dir_name + "/genfile.gen"
        inp_fname = self.minimize_template
        out_fname = dir_name+"/output.out"
        out_coords_fname = dir_name+"/geom.out.xyz"

        struct_dict = {"coords": coords,
                       "names": self.particle_names,
                       "types": self.particle_types,
                       "Natoms": self.natoms}

        self.write_gen(inp_gen_fname, struct_dict)

        with open(out_fname, "w") as outf:
            commands = [self.run_string, inp_fname]
            dftb_process = subprocess.Popen(commands, cwd=dir_name, shell=True, stdout=outf, stderr=outf)

        exit_code = dftb_process.wait()


        if exit_code != 0:
            try:
                raise DFTBError(f"DFTB+ exited unexpectedly with exitcode: {exit_code}\n")
            except DFTBError as error:
                self.log.exception(error)
                raise
        else:
            self.log.info(f"DFTB+ exited successfully. Exit code: {exit_code}")

        energy, coords = self.parse_dftbp_output(out_fname, out_coords_fname, **kwargs)

        # print(f"\n\nEnergy: {energy},\n\n coords:\n{coords}\n\nlabels:\n{atom_labels}\n\n")


        cluster.set_particle_positions((coords, molecule_ids, atom_labels))

        cluster.cost = float(energy)

        return cluster

    def get_energy(self, cluster: Cluster, *args, **kwargs) -> float:
        """
        Method to get the energy of a geometry
        """
        raise NotImplementedError


class TestDftbPotential(TestCase):
    """Tests for the DFTB+ interface potential"""
    def setUp(self) -> None:
        """Sets up the test directory path"""
        self.test_data_path = os.path.abspath("../tests/test_data/")

    def test_parser_good(self):

        energy, coords = DFTBPotential.parse_dftbp_output(self.test_data_path + "/dftb_good_output.test",
                                                          out_coords_fname=self.test_data_path+"/dftb_good_geom.xyz")

        self.assertListEqual(list(coords[-1]), [1.150167,   -0.16129964,  0.13635255])

        self.assertEqual(energy, -2448.9224)

    def test_parser_no_struct(self):

        with self.assertRaises(IOError):
            DFTBPotential.parse_dftbp_output(self.test_data_path + "/dftb_good_output.test",
                                             out_coords_fname=self.test_data_path+"/dftb_bad_geom.xyz")

    def test_parser_unminimised_output(self):

        with self.assertRaises(DFTBError):
            DFTBPotential.parse_dftbp_output(self.test_data_path + "/dftb_bad_output.test",
                                             out_coords_fname=self.test_data_path+"/dftb_good_geom.xyz")

    def test_dftb_exited_unsuccessfully(self):

        raise self.skipTest("Test not implemented")

    def test_minimise(self):

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

        pot = DFTBPotential(minimize_template=self.test_data_path+"/dftb_in.hsd", energy_template="NONE", work_dir=".",
                            run_string="/home/john/.local/bin//dftb+")
        clus2 = pot.minimize(clus1, dir_name=testdir)

        print(clus2.cost)

        coords, mol_ids, atom_names = clus2.get_particle_positions()

        self.assertListEqual(atom_names, ["H", "H", "O", "H", "H", "O"])

        if os.path.exists(testdir):
            shutil.rmtree(testdir)
