import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import unittest
from uuid import uuid4
import os
import logging

from glob import glob

from Sophie import Atom, parse_mol_file
from analyse_cycles import GraphCycleAnalyser, plot_labelled_graph

G=nx.Graph()
# nx.get_node_attributes
G = nx.OrderedGraph()


class Alcohol:
    from Sophie import Atom, parse_mol_file
    for atom in G.nodes():
        if atom.element == "O":
            for neighbor in G.neighbors(atom):
                if neighbor.element == "H":
                    # print("Alcohol")
                else:
                    pass
        else:
            pass


