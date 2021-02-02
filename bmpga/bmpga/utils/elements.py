# coding=utf-8
"""Provides some useful utilities for dealing with various physical properties of chemical systems"""
import os
import bmpga
import logging

import pandas as pd

from string import digits
from functools import lru_cache


class ParticleRadii(object):
    """
    Class to lookup and return atomic radii.
    Uses radii collated by crystalmaker by default.
    See bmpga.data.particle_radii.csv for more info

    """
    def __init__(self, csv_path: str or None=None, overrides: list or None=None, logger=False) -> None:

        self.data_path = bmpga.__path__[0] + "/data/particle_radii.csv"  # set default data path
        self.logger = logger or logging.getLogger(__name__)

        if csv_path is not None:
            if os.path.isfile(csv_path):
                self.data_path = csv_path
            else:
                raise OSError("File: {} does not exist!!".format(csv_path))

        self.particle_radii = pd.read_csv(self.data_path, comment='#', index_col=1)

        self._atomic_number_dict = self._get_atomic_number_mapping()

        # Overrides must have format: list([[str(Symbol)/int(Number), str(radius type), float(value)], ...])
        if overrides is not None:
            for override_radius in overrides:
                if isinstance(override_radius[0], int):
                    label = self._atomic_number_dict[override_radius[0]]

                    self.logger.info(("Updating {}/{}.{} to {}"
                                      .format(override_radius[0], label, override_radius[1], override_radius[2])))

                    self.particle_radii.at[label, override_radius[1]] = override_radius[2]

                elif isinstance(override_radius[0], str):

                    self.logger.info("Updating {}.{} to {}"
                                     .format(override_radius[0], override_radius[1], override_radius[2]))

                    self.particle_radii.at[[override_radius[0]], override_radius[1]] = override_radius[2]

    def _get_atomic_number_mapping(self) -> dict:
        """Inelegant private method to create a mapping between atomic number and symbol"""
        part_dict = dict(self.particle_radii.loc[:, "Number"])
        return dict(zip(part_dict.values(), part_dict.keys()))

    def __call__(self, element: str or int, radius_type: str or None=None) -> float:
        return self.get_radius(element)

    def get_radius(self, symbol: str or int, user_radius: str or None=None) -> float:
        """Returns the radius of the requested particle. Can request by name or atomic number

        Tries to return radii from memory in this order:
            Covalent > Atomic > Ionic > VdW > Crystal

        User can request a specific radius to be returned if they wish

        Args:
            symbol: str or int, required, The label or atomic number of the particle to be checked
            user_radius: str, optional, the specific type of radius required from the options above (default=None)

        Returns:
            float, particle radius in Angstrom

        Raises:
            ValueError: If no radius is found
            KeyError: If particle label/atomic number is not in memory.
                       See overrides in self.__init__ for how to correct this

        """
        # This will return the first radius that is associated with a particle symbol.
        # Will print a warning (suppressed by self.silent=True) if the returned radius is not the covalent radius

        preferred_order = ['Covalent', 'Atomic', 'Ionic', 'VdW', 'Crystal']

        if type(symbol) is int:
            symbol = self._atomic_number_dict[symbol]

        if user_radius is not None:
            radius = self.particle_radii.loc[symbol, user_radius]
            if radius > 0.0:
                return radius

            else:
                message = "{} radius not found for {}\n".format(user_radius, symbol)
                try:
                    raise ValueError(message)
                except ValueError:
                    self.logger.exception(message)
                    raise

        try:
            symbol = translate_symbol(symbol)
            radius = self.particle_radii.loc[symbol, 'Covalent']

        except KeyError:
            message = "No data found for {}. Check to ensure it is listed in {}\n".format(symbol, self.data_path)
            self.logger.exception(message)
            raise KeyError(message)

        if radius >= 0.0:  # Missing values in a DataFrame are nan. nan < 0.0 and radii can't be negative...
            return radius

        warn = "Warning! Covalent radius not found for {}: ".format(symbol)
        for rad_type in preferred_order:
            radius = self.particle_radii.loc[symbol, rad_type]
            if radius >= 0.0:  # Missing values in a DataFrame are nan. nan < 0.0 and radii can't be negative...
                warn += "using {} radius ({}) instead!\n".format(rad_type, radius)
                self.logger.warning(warn)
                return radius

        error_message = "No radius found for {}\n".format(symbol)

        try:
            raise ValueError(error_message)
        except ValueError as error:
            self.logger.exception(error_message)
            raise error


@lru_cache(maxsize=None)
def translate_symbol(symbol: str) -> str:
    """Removes numbers from symbols"""

    if symbol in ["EPW"]:
        return symbol

    # Strip numbers out of the symbol
    remove_digits = str.maketrans("", "", digits)
    symbol = symbol.translate(remove_digits)

    # print(symbol)
    # if len(symbol) > 1:
    #     if symbol[1].isupper():
    #         symbol = symbol[0]
    #
    # ele_dict = {"OXT": "O",
    #             "HA": "H",
    #             "HB": "H",
    #             "HC": "H",
    #             "HE": "H",
    #             "HG": "H",
    #             "HZ": "H",
    #             "CA": "C",
    #             "CB": "C",
    #             "CC": "C",
    #             "CD": "C",
    #             "CG": "C",
    #             "NA": "N",
    #             "NB": "N",
    #             "NCD": "N",
    #             "NCG": "N",
    #             "E": "EPW"}
    #
    # if symbol in ele_dict.keys():
    #     return ele_dict[symbol]
    # else:
    return symbol
