import networkx as nx
import numpy as np
import networkx.algorithms.isomorphism as iso

# def edge_match(G, cycles):
#     cycle1 =
#     em = iso.numerical_edge_match('weight', 1)
#     nx.is_isomorphic(cycle1, cycle2)
# https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.isomorphism.is_isomorphic.html#networkx.algorithms.isomorphism.is_isomorphic

# noinspection PyPep8Naming,PyUnusedLocal
def count_atoms(G, cycle):
    return len(cycle)


# noinspection PyPep8Naming,PyUnusedLocal
def count_carbons(G, cycle):
    counter = 0
    for i, atom in enumerate(cycle):
        if atom.element == "C":
            counter += 1
    return counter


def count_boron(G, cycle):
    counter = 0
    for i, atom in enumerate(cycle):
        if atom.element == "B":
            counter += 1
    return counter


# noinspection PyPep8Naming,PyUnusedLocal
def count_oxygen(G, cycle):
    counter = 0
    for i, atom in enumerate(cycle):
        if atom.element == "O":
            counter += 1
    return counter


# noinspection PyPep8Naming,PyUnusedLocal
def count_nitrogen(G, cycle):
    counter = 0
    for i, atom in enumerate(cycle):
        if atom.element == "N":
            counter += 1
    return counter


# noinspection PyPep8Naming,PyUnusedLocal
def count_sulphur(G, cycle):
    counter = 0
    for i, atom in enumerate(cycle):
        if atom.element == "S":
            counter += 1
    return counter

def get_edge_weight(G, N1, N2):
    for e in G.edges().data(nbunch=N1):
        if e[1] == N2:
            return e[2]["weight"]

def set_2_cycle(G, cycle):
    """Horrible hack to reconstruct cycles from sets (unordered -> ordered)"""
    unordered_cycle = list(cycle)
    current_atom = unordered_cycle.pop()

    ordered_cycle = [current_atom]

    counter = 0
    while len(unordered_cycle) >= 1:  
        counter += 1
        for n in G.neighbors(current_atom):

            if n in unordered_cycle:
                idx = unordered_cycle.index(n)
                current_atom = unordered_cycle.pop(idx)
                ordered_cycle.append(current_atom)                
                break
            
            else:
                pass

            if len(cycle) == 12:
                print(f"{[a.element for a in unordered_cycle]} -> {[a.element for a in ordered_cycle]}")
                
        if counter > len(cycle)**2:
            message = f"Failed to order cycle: {[a.element for a in cycle]}"
            # print(message)
            cycle = list(cycle)
            np.random.shuffle(cycle)
            return set_2_cycle(G, cycle)
            #raise ValueError(message)
        # # print(unordered_cycle, ordered_cycle)
    return ordered_cycle


def get_cycle_edge_weights(G, cycle):
    edge_weights = []
    cycle = set_2_cycle(G, cycle)
    for idx, N1 in enumerate(cycle[:-1]):  # iterate through and add edge weights
        N2 = cycle[idx + 1]
        edge_weights.append(get_edge_weight(G, N1, N2))
    edge_weights.append(get_edge_weight(G, cycle[0], cycle[-1]))  # add the weight for the edge between first and last node of cycle

    return edge_weights


def analysis_of_rings(G, cycles):
    semantic_definitions = []

    analysis_methods = [
        count_atoms,
        count_carbons,
        count_oxygen,
        count_nitrogen,
        count_sulphur,
#        get_cycle_edge_weights,
    ]

    for cycle in cycles:
        cycle_semantics = []
        for method in analysis_methods:
            cycle_semantics.append(method(G, cycle))
        #print(f"atoms = {cycle_semantics[0]}", f"carbon = {cycle_semantics[1]}",
        #      f"oxygen = {cycle_semantics[2]}", f"nitrogen = {cycle_semantics[3]}", f"sulphur = {cycle_semantics[4]}",)
#              f"double_bonds = {cycle_semantics[5]}" )
        semantic_definitions.append(cycle_semantics)
        
        # # print(f"cycle_edge_weights = {semantic_definitions[0]}")

    return semantic_definitions


# def create_subgraph(G, cycles):
#     for cycle in cycles:
#
#     H = G.subgraph([(0, 1), (3, 4)])
# G.subgraph(nodes).copy()

# if __name__ == "__main__":
#     cycle = ["a", "b", "c", "d"]
#
#     G = nx.Graph()
#
#     G.add_nodes_from(cycle)
#
#     G.add_weighted_edges_from([("a", "b", 1), ("b", "c", 2), ("c", "d", 3), ("d", "a", 4), ])
#
#     edge_weights = get_cycle_edge_weights(G, cycle)
#
#     # print(edge_weights)  # should # print [1,2,3,4]
#
# # print(count_atoms(Graph, allcycles))

