# coding=utf-8
"""
This module provides the basic Mutate object.
This object is initialised with the list of mutations you wish to perform and their relative probabilities.
To call this object, simply invoke the mutate() method with a Cluster instance as the argument, this will return
   a new Cluster instance with a mutation applied.

"""
import copy
import logging
from typing import List

import numpy as np

from bmpga.mutation import BaseMutation, RandomSingleTranslation
from bmpga.storage import Cluster


class Mutate(object):
    """Main mutation class.

    This object is initialised with the list of mutations you wish to perform and their relative probabilities.
    To call this object, simply invoke the mutate() method with a Cluster instance as the argument, this will return
        a new Cluster instance with a mutation applied.
    """

    def __init__(self,
                 mutations: List[BaseMutation]=None,
                 relative_probabilities: List[float]=None,
                 log: logging.Logger = logging.getLogger(__name__)) -> None:
        """

        Args:
            mutations (List[BaseMutation]): required, This is a list of instances of the
                    mutations you wish to use. (default=RandomSingleTranslation())
            relative_probabilities (List[float]): A list of the relative probabilities for the mutation instances you
                    have provided. This must be of the same length as the previous argument. (default=np.ones())
            log (logging.Logger()): Pass this if you wish to override the default logger.
        """
        self.log = log

        if mutations is None:
            self.log.warning("No mutations passed to Mutate. Using only single molecule translations!")
            self.mutations = [RandomSingleTranslation()]
            self.probabilities = [1]

        elif mutations is not None:
            self.mutations = mutations

        # Check and set the mutation probabilities. (Also normalizes them)
        if relative_probabilities is not None:
            try:
                assert len(relative_probabilities) == len(self.mutations)
            except AssertionError as error:
                self.log.exception("Probabilities must be the same length as mutations, or None\n{}".format(error))
                raise
            self.probabilities = relative_probabilities / np.sum(relative_probabilities)
        else:
            relative_probabilities = np.ones(shape=np.shape(self.mutations))
            self.probabilities = relative_probabilities / np.sum(relative_probabilities)

    def mutate(self, cluster: Cluster = None) -> Cluster:

        mutation = np.random.choice(self.mutations, p=self.probabilities)
        self.log.debug(f"Performing {mutation.__class__.__name__} mutation on: {cluster}")
        # mutated_cluster = Cluster(cost=0.0, molecules=copy_molecules(cluster.molecules))
        # self.log.debug(f"Mutated cluster = {mutated_cluster}")
        parent = copy.deepcopy(cluster)
        parent.public_id = None
        parent.cost = 0.0
        return mutation(parent)  # Mutation.call() returns a cluster object
