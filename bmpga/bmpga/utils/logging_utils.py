# coding=utf-8
"""
Sets up project level logging using the logging module
from then python standard library.

Config files are stored in the bmpga.data directory.
logger_dev.json logs everything including debug info to stdout and to a log file,
      this is probably overkill for normal users
logger_prod.json logs everything greater than INFO to stdout so users can capture it


Examples:
    >>> import logging
    >>> from bmpga.utils.logging_utils import setup_logging
    >>> setup_logging()
    >>> logger = logging.getLogger(__name__)


"""
import bmpga
import json
import os

import logging.config


def setup_logging(config_path: str=bmpga.__path__[0] + "/data/logger_dev.json",
                  default_logging_level=logging.INFO,
                  env_key='LOG_CFG') -> None:

    """
    Setup logging configuration

    """
    path = config_path
    value = os.getenv(env_key)

    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)

    else:
        logging.basicConfig(level=default_logging_level)
