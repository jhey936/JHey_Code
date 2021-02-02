import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from uuid import uuid4
from natural_products.tree_analysis import add_element, tree_creation
from glob import glob

from natural_products.analyse_cycles import GraphCycleAnalyser, plot_labelled_graph
from natural_products.cycleseparation import analysis_of_rings

class Atom:
    def __init__(self, element, atom_number, coordinate=np.zeros(3), data=dict()):
        self.element = element
        self.atom_number = atom_number
        self.data = data
        self.coordinate = coordinate
        self.id = uuid4().int

    def __hash__(self):
        return self.id

    def __str__(self):
        return str(self.element)+str(self.atom_number)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):

        return self.atom_number < other.atom_number


def parse_mol_file(file_name, show_graph=False, save_graph=False):

    with open(file_name, "r") as f:
        # print(file_name)
        i = 0
        total_num_atoms = 0
        total_num_bonds = 0
        G = nx.OrderedGraph()
        atoms = []

        for line in f: 
            if i == 3:

                total_num_atoms = int(line[0:3])
                # position 0 in the line is the number of atoms
                total_num_bonds = int(line[3:6])
                # position 1 in the line is the number of bonds, this is different to the number of atoms in some cases
                #I initially was goingt to create a matrix and then a graph, but from the molfiles I am able to go straight to a graph so I figured why not
            elif i > 3 and i < total_num_atoms + 4:
                # the above line sets the code so that it starts from line 4 and ends at whatever line the total number of atoms are + the 4 lines at the top that are irrelevant
                l = line.split()
                elm = l[3]
                #Position three in the line has the Atoms listed so this plucks the element out which can then be added to an array
                atom = Atom(elm, atom_number=i-3, coordinate=[l[0],l[1],l[2]])
                atoms.append(atom)
                G.add_node(atom)
                # if G.node [elm] = "O":
                #     if g.neighbor == "H"


            elif i >= total_num_atoms + 4 and i < total_num_atoms + total_num_bonds + 4:
                # The above line starts off the creation of the graph by setting it to the correct line in the mol file.
                start_elem = int(line[0:3])
                # The above. takes the number of the first atom and the one below takes the number of the second atom
                end_elem = int(line[3:6])
                bond_weight = int(line[6:9])
                chirality = int(line[9:12])
                # The bond weight is whether its a double, single or triple bond and it is given as an edge weighting 
            
                G.add_weighted_edges_from([(atoms[start_elem-1], atoms[end_elem-1], bond_weight)])
            i+=1

    analyser = GraphCycleAnalyser(G, verbose=False)
    cycles = analyser.find_all_cycles()
    # for G.node in G.nodes():
    #     if G.node.elm = "O"
    # # print([[str(i)for i in c] for c in cycles], len(cycles))

    if show_graph:
        plot_labelled_graph(G)
        plt.show(block=True)  
        if save_graph:
            plt.savefig(file_name.split(".")[0]+".png")
    else:
        pass
    return G, cycles


class Counter:

    def __init__(self):

        self.dict = {}

    def add_element(self, element):

        element = ".".join([str(i) for i in element])
        if element in self.dict.keys():

            self.dict[element] = self.dict[element]+1

        else:
            self.dict[element] = 1

    def __call__(self, element):
        self.add_element(element)

