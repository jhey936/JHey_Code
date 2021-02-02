# coding=utf-8
"""Provides the parent class other characterizers should inherit from"""
import logging

from bmpga.storage import Cluster


class BaseCharacterizer(object):
    # TODO: Implement and document BaseCharacterizer
    """Base class for structural characterisation."""
    def __init__(self, log=None) -> None:
        self.log = log or logging.getLogger(__name__)

    def compare(self, cluster1: Cluster, cluster2: Cluster, *args, **kwargs) -> bool:
        """
        
        Args:
            cluster1: 
            cluster2: 
            *args: 
            **kwargs: 

        Returns:

        """
        raise NotImplementedError

    def __call__(self, cluster1: Cluster, cluster2: Cluster, *args, **kwargs) -> bool:
        return self.compare(cluster1, cluster2, *args, **kwargs)
