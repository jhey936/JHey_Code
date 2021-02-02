# coding=utf-8
"""The base DFT potential class. Other DFT potentials should inherit from this class"""
import os
import logging

import numpy as np

from typing import List

from bmpga.storage import Cluster

from bmpga.potentials.base_potential import BasePotential


class BaseDFTPotential(BasePotential):
    """The base DFT potential class. Other DFT potentials should inherit from this class"""
    def __init__(self, work_dir: str, run_string: str, log: logging.Logger=logging.getLogger(__name__), **kwargs):
        self.work_dir = work_dir
        self.run_string = run_string
        self.log = log
        self.last_dir = 0
        super().__init__(**kwargs)

    def get_directory(self, dirname="ABIN") -> str:
        """Finds the next work directory """
        while True:

            self.last_dir += 1

            try:
                name = f"{dirname}{self.last_dir}"
                if not os.path.exists(self.work_dir + name):
                    self.log.info(f"Making {name}")
                    os.makedirs(f"{self.work_dir}" + name)
                    return name
            except FileExistsError:
                continue

    def get_energy(self, cluster: Cluster, *args, **kwargs) -> float:
        """Overload with a method to write, submit and parse a dft input file to get energy

        Args:
            cluster (Cluster): required, the cluster to be minimised
        """
        raise NotImplementedError

    def minimize(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Overload with a method to write, submit and parse a dft input file to minimise and return a cluster

        Args:
            cluster (Cluster): required, the cluster to be minimised

        """
        raise NotImplementedError

    # noinspection PyPep8Naming
    @staticmethod
    def format_XYZ(coordinates: np.ndarray, labels: List[str]) -> str:
        """Formats a list of coordinates and labels into xyz format"""

        # noinspection PyPep8Naming
        XYZ_string = ""
        for label, coordinate in zip(labels, coordinates):
            XYZ_string += f"{label}    {coordinate[0]}    {coordinate[1]}    {coordinate[2]}\n"
        return XYZ_string

    def insert_to_template(self,
                           template: str,
                           out_file: str,
                           target_dict: dict,
                           ) -> None:
        """replaces XML style tags (targets) in a template with the supplied values

        Args:
            target_dict:
            template:
            out_file:

        Returns:
            None

        """
        try:
            with open(template) as template_f:
                template_data = template_f.readlines()
        except IOError as error:
            message = f"Error reading {template}.\n{error}\n"
            self.log.exception(message)
            raise

        formatted_data = []
        for line in template_data:
            for key in target_dict.keys():
                if key in line:
                    line = line.replace(key, target_dict[key])
            formatted_data.append(line)

        try:
            with open(out_file, "w+") as out_f:
                out_f.writelines(formatted_data)
        except IOError as error:
            message = f"""Error writing to {out_file}, does path exist and do you have the correct permissions?" \
                      {error}\n"""
            self.log.exception(message)
            raise
