import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import unittest

from typing import List

from uuid import uuid4
import os
import logging

from glob import glob

from Sophie import Atom, parse_mol_file
from analyse_cycles import GraphCycleAnalyser, plot_labelled_graph

from Sophie import Atom, parse_mol_file

def IdentifyAlcohols(G:nx.Graph) -> List:

    alcohols = []
    for atom in G.nodes():
        if atom.element == "O":
            for neighbour in G.neighbors(atom):
                if neighbour.element == "H":
                    alcohols.append([atom, neighbour])
                    # print(f"Alcohol: {str(atom), str(neighbour)}")
                else:
                    pass
        else:
            continue

    return alcohols

def IdentifyEthers(G:nx.Graph) -> List:

    ethers = []
    for atom in G.nodes():
        if atom.element == "C":
            for neighbour in G.neighbors(atom):
                if neighbour.element == "O":
                    if neighbour.element == "C":
                        ethers.append([atom, neighbour])
                        # print(f"Ether: {str(atom), str(neighbour)}")
                    else:
                        pass                
                else:
                    pass
        else:
            continue

    return ethers

def IdentifySulfides(G:nx.Graph) -> List:

    sulfides = []
    for atom in G.nodes():
        if atom.element == "C":
            for neighbour in G.neighbors(atom):
                if neighbour.element == "S":
                    if neighbour.element == "C":
                        sulfides.append([atom, neighbour])
                        # print(f"Sulfide: {str(atom), str(neighbour)}")
                    else:
                        pass                
                else:
                    pass
        else:
            continue

    return sulfides


if __name__ == "__main__":
    G, cycles = parse_mol_file("NPC100767.mol")

    alcohol_groups = IdentifyAlcohols(G)
    ether_groups = IdentifyEthers(G)
    sulfide_groups = IdentifySulfides
