# coding=utf-8
"""
========================================================
The Birmingham Molecular Pool Genetic Algorithm (BMPGA)
John C. Hey 
========================================================

A Genetic Algorithm (GA) for the global optimisation of molecular clusters, 
and energy landscape exploration using the Threshold Algorithm (Schoen1995)

"""
from bmpga.utils.logging_utils import setup_logging
setup_logging(__path__[0] + "/data/logger_dev.json")
# setup_logging(__path__[0] + "/data/logger_prod.json")

__version__ = 0.2
