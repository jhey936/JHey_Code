# coding=utf8
"""NWChem parser"""
import os
import logging
import subprocess

import numpy as np
import pandas as pd

from bmpga.storage import Cluster
from bmpga.errors import DFTExitedUnexpectedlyError

from bmpga.storage.molecule import IdentifyFragments
from bmpga.potentials.dft import BaseDFTPotential


class NWChemPotential(BaseDFTPotential):

    def __init__(self,
                 minimize_template: str,
                 energy_template: str,
                 work_dir: str,
                 run_string: str = f"mpiexec -n 2 nwchem",  # TODO: remove this after testing
                 log: logging.Logger = logging.getLogger(__name__)):

        self.minimize_template = minimize_template
        self.energy_template = energy_template

        self.log = log

        if work_dir[-1] != "/":
            work_dir += "/"

        super().__init__(work_dir=work_dir, run_string=run_string)

# self.last_dir = 0

    def get_directory(self) -> str:

        while True:

            self.last_dir += 1

            try:
                name = f"NWCHEM{self.last_dir}"
                if not os.path.exists(self.work_dir + name):
                    self.log.info(f"Making {name}")
                    os.makedirs(f"{self.work_dir}" + name)
                    return name
            except FileExistsError:
                continue

    def minimize(self, cluster: Cluster, *args, **kwargs) -> Cluster:

        name = self.get_directory()

        raw_coordinates = cluster.get_particle_positions()
        coordinates = self.format_XYZ(raw_coordinates[0], raw_coordinates[2])

        tag_dict = {"<XYZ>": coordinates, "<NAME>": name}

        run_dir = self.work_dir + name

        input_file_name = run_dir + "/input.nw"
        output_file_name = run_dir + "/output.out"

        self.insert_to_template(template=self.minimize_template, out_file=input_file_name, target_dict=tag_dict)

        commands = [self.run_string, f"{input_file_name}", f"> {output_file_name}"]
        commands = ' '.join(commands)

        self.log.info(f"Starting NWChem with commands: {commands}")

        nwchem_process = subprocess.Popen(commands, cwd=run_dir, shell=True)

        exit_code = nwchem_process.wait()

        if exit_code == 134:
            try:
                raise DFTExitedUnexpectedlyError(f"NWChem exited unexpectedly with exitcode: {exit_code}\n")
            except DFTExitedUnexpectedlyError as error:
                self.log.exception(error)
                raise

        if exit_code != 0:
            try:
                raise DFTExitedUnexpectedlyError(f"NWChem exited unexpectedly with exitcode: {exit_code}\n")
            except DFTExitedUnexpectedlyError as error:
                self.log.exception(error)
                raise
        else:
            self.log.info(f"NWChem exited successfully. Exit code: {exit_code}")

        output_parser = NWChemOutputParser(output_file_name)

        output_parser.parse()

        final_structure = output_parser.final_structure

        try:
            assert all(raw_coordinates[2]) == all(final_structure[0])
        except AssertionError as error:
            self.log.exception(f"Atoms out of order! {raw_coordinates[2]} != {final_structure[0]}\n{error}")
        cluster.set_particle_positions((final_structure[1], raw_coordinates[1], final_structure[0]))

        return cluster

    def get_energy(self, cluster: Cluster, *args, **kwargs):
        try:
            raise NotImplementedError("This method has not been implemented yet")
        except NotImplementedError as error:
            self.log.exception(error)
            raise


class NWChemOutputParser(object):
    """
    Class to parse NWChem output files to provide a cluster object of the re-minimized
    structure with all the various calculated properties in the object's properties dict.
    """

    def __init__(self, file_name) -> None:
        self.filename = file_name
        self.data = []
        self.extra_info = dict()
        self.extra_info["ENERGIES"] = []
        self.final_structure = []
        self.frequencies = []
        self.n_atoms = 0

        self.log = logging.getLogger(__name__)

    def read(self) -> None:
        with open(self.filename) as f:
            raw_data = f.readlines()

        for line in raw_data:
            line = line.rstrip()
            line = line.lstrip()
            if line != '':
                self.data.append(line)

    def parse(self) -> None:
        self.read()
        self.check_completed()

        self.get_input_deck(self.data)

        self.get_moments_of_inertia()

        if "INPUT" in self.extra_info.keys():
            if "TASK DFT FREQ" in self.extra_info["INPUT"]:
                self.get_frequencies()
        else:
            for idx, line in enumerate(self.data):
                if "XYZ format geometry" in line:
                    self.n_atoms = int(self.data[idx + 2].strip())
                    break

        self.get_structs()
        self.get_energies()

    def get_energies(self) -> None:
        for line in self.data:
            if line[0] == "@":
                line = line.split()
                try:
                    energy = float(line[2])
                    self.extra_info["ENERGIES"].append(energy)
                    continue
                except ValueError:
                    pass
        # noinspection PyAttributeOutsideInit
        self.final_energy = self.extra_info["ENERGIES"][0]

    def get_moments_of_inertia(self) -> None:
        moments = []
        for idx, line in enumerate(self.data):
            if line == "moments of inertia (a.u.)":
                array = self.data[idx + 2: idx + 5]
                for i, row in enumerate(array):
                    row = row.split()
                    row = np.array([float(v) for v in row])
                    array[i] = row
                moments.append(array)
        self.extra_info["ALL_INERTIA"] = moments
        # noinspection PyAttributeOutsideInit
        self.final_inertia = np.array(moments[-1])

    def check_completed(self) -> None:
        try:
            self.data.index("CITATION")
        except ValueError:
            raise ValueError("DID NOT COMPLETE")

    def get_input_deck(self, file_data) -> None:
        """Returns a copy of the input deck if present, logs an error and """
        try:
            start = file_data.index("============================== echo of input deck ==============================")

        except ValueError:
            self.log.info("echo of input deck not found in {}".format(self.filename))
            return None

        try:
            end = file_data.index("================================================================================")
        except ValueError:
            self.log.info("end of input deck not found in {}".format(self.filename))
            return None

        input_deck = file_data[start:end]

        for i, line in enumerate(input_deck):
            try:
                line = line.split(" ")
            except TypeError:
                pass
            if type(line) == list:
                if line[0] == "GEOMETRY":
                    end = input_deck[i:].index("END")
                    # noinspection PyAttributeOutsideInit
                    self.initial_structure = input_deck[i + 1:i + end]
                    self.n_atoms = len(self.initial_structure)

        self.extra_info["input_deck"] = input_deck

    def get_frequencies(self) -> None:
        """Parses the NWChem output for the results of the normal mode analysis"""
        try:
            start = self.data.index("Normal Eigenvalue ||           Projected Infra Red Intensities")
        except ValueError:
            raise ValueError("No Freq Output")

        raw_frequencies = self.data[start + 3: start + 3 + (3 * self.n_atoms)]

        for i, line in enumerate(raw_frequencies):
            line = line.split(" ")
            line = [v for v in line if v != "" and v != "||"]
            line = {"Mode": int(line[0]), "Freq": float(line[1]), "I": float(line[5])}
            raw_frequencies[i] = line

        self.frequencies = pd.DataFrame(raw_frequencies)
        self.frequencies.set_index("Mode", inplace=True)

    def get_structs(self) -> None:
        """
        Method to parse all the structures out of an NWChem output file and
        return them as self.extra_info["QUENCH_STRUCTS"]

        Also returns the final minimised structure as self.final_structure
        """
        all_structs = []
        for idx, line in enumerate(self.data):
            if line == 'Geometry "geometry" -> "geometry"':
                offset = idx+5   # This may be specific to versions of
                coords = [string.split(" ") for string in self.data[offset: offset + self.n_atoms]]

                coords = [[string for string in sublist if string != ''] for sublist in coords]

                coords = np.array(coords)

                labels = coords[:, 1]
                coords = coords[:, 2:]

                structure = [labels, np.array(coords, dtype=np.float64)]

                all_structs.append(structure)

        self.extra_info["ALL_STRUCTS"] = all_structs
        self.final_structure = all_structs[-1]
