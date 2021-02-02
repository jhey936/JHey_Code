import networkx as nx
from natural_products.io_utils import Atom
import copy
# import itertools
# from natural_products.cycleseparation import analysis_of_rings


def join_set(l):
    _l = ["".join(i) for i in l]
    return set(_l)


def get_ordered_path(G):

    start_node, end_node = 0, 0

    for n in G.nodes():
        if nx.degree(G, n) == 2:
            start_node = n

            for n2 in G.neighbors(n):

                if nx.degree(G, n2) == 2:
                    end_node = n2
                    break


    last_edge_weight = G.get_edge_data(start_node, end_node)

    try:
        G.remove_edge(start_node, end_node)
    except nx.NetworkXError:
        with open("failed_smiles.error", "a") as EF:
            EF.write("@?@?@?@?@?@?@?@@?@?@?@?@?@@?\n")
            EF.write(",".join([str(n) for n in G.nodes]))
            EF.write(f"{start_node}, {end_node}\n")
        raise ValueError

    attempts = 0
    max_tries = len(G.nodes)

    while len(nx.shortest_path(G, start_node, end_node)) < len(G.nodes):
        
        if attempts >= max_tries:
            error_mes = f"Could not find a shortest path for Graph {[str(n) for n in G.nodes]}"
            with open("failed_smiles.error", "a") as EF:
                EF.write("====================\n")
                EF.write(error_mes)
                EF.write("\n")
            raise ValueError(error_mes)
        
        attempts+=1
        
        current_shortest_path = nx.shortest_path(G, start_node, end_node)

        for node in current_shortest_path:
            if nx.degree(G, node) > 2:

                neighbors = G.neighbors(node)

                for n2 in [n for n in neighbors if nx.degree(G, n) > 2]:
                    temp_G = copy.deepcopy(G)

                    # print("removing edge: ", node, n2)
                    temp_G.remove_edge(node, n2)

                    if len(nx.shortest_path(temp_G, start_node, end_node)) > len(current_shortest_path):

                        G = copy.deepcopy(temp_G)
                        current_shortest_path = nx.shortest_path(G, start_node, end_node)

                        break
                    else:
                        temp_G = copy.deepcopy(G)

    G.add_edge(start_node, end_node, **last_edge_weight)
    return G


def smiles(G, cycle):
    # number_of_elements = len(cycle)
    edge_weight_dict = {1: "-", 2: "=", 3: "≡"}

    list_of_smiles = []

    cycle_G = nx.Graph(nx.subgraph(G, cycle))

    cycle_graph = get_ordered_path(cycle_G)

    total_nodes = cycle_graph.number_of_nodes()

    # ring_analysis = analysis_of_rings(G,cycle)
    # total_no = ring_analysis[0]

    nodes = [n for n in cycle_graph.nodes]

    first_node = nodes[0]
    list_of_edges = []

    last_node = list(nx.neighbors(cycle_graph, first_node))[0]

    last_edge_weight = cycle_graph.get_edge_data(first_node, last_node)["weight"]

    cycle_graph.remove_edge(first_node, last_node)

    path = nx.shortest_path(cycle_graph, first_node, last_node)

    for N1, N2 in zip(path[:-1], path[1:]):
        list_of_smiles.append(N1.element)

        edge_weight = cycle_graph.get_edge_data(N1, N2)["weight"]
        list_of_smiles.append(edge_weight_dict[edge_weight])

    list_of_smiles.append(N2.element)
    list_of_smiles.append(edge_weight_dict[last_edge_weight])
    # if len(list_of_smiles/2) < total_nodes:
    #     for n in nodes:
    #         if n.neighbors() == 3:
    #             last_edge = cycle_graph.get_edge_data(n, n.neighbor)["weight"]
    #             cycle_graph.remove_edge(n, n.neighbor)
    #             path = nx.shortest_path(cycle_graph, n, n.neighbor)
    #             if len(path)+1 < total_nodes:
    #                 cycle_graph.append(last_edge)
    #             else:
    #                 for N1, N2 in zip(path[:-1], path[1:]):
    #                     list_of_smiles.append(N1.element)
    #
    #                     edge_weight = cycle_graph.get_edge_data(N1, N2)["weight"]
    #                     list_of_smiles.append(edge_weight_dict[edge_weight])
    #
    #                 list_of_smiles.append(N2.element)
    #                 list_of_smiles.append(edge_weight_dict[last_edge])
    #
    #             #list_of_edges.append((n, n+1)) # is this adding the nodes as a tuple
    #     for i in range (0, len(list_of_edges)+1):
    #         for subset in itertools.permutations(list_of_edges, i):
    #             for l in subset:
    #                 cycle_graph.remove_edge(list_of_edges[0], list_of_edges[1]) #need to test it for the first result oof
    #



    # if len(list_of_smiles/2) != total_no:

    smiles_variation = total_smiles(list_of_smiles)

    return smiles_variation


def total_smiles(list_of_smiles):

    all_smiles = [list_of_smiles]
    
    for i in range(int(len(list_of_smiles)/2)):

        tmp = copy.deepcopy(all_smiles[-1])

        tmp.append(tmp.pop(0))
        tmp.append(tmp.pop(0))

        all_smiles.append(copy.deepcopy(tmp))

        rev_tmp = copy.deepcopy(tmp)
        rev_tmp.append(tmp.pop(0))
        rev_tmp.reverse()
        all_smiles.append(copy.deepcopy(rev_tmp))

    all_smiles.pop()
    return all_smiles
    # list(permutations(l))
    # G.get_edge_data(1, 2)["weight"]
#
# mylist = ['a', 'b', 'c', 'd', 'e']
# myorder = [3, 2, 0, 1, 4]
# mylist = [mylist[i] for i in myorder]
# # print(mylist)
#
#     for node in cycle.nodes:
#         list_of_smiles.append(node)
#         for neighbor in cycle.neighbors:
#             list_of_smiles.append(edge[0])
#             cycle.remove_edge(edge[0])
#             shortest_path=nx.shortest_path(node, neighbor)
#
#             # list_of_smiles.append(shortest_path)
#             nx.write_weighted_edgelist(cycle)
#
# for atom in G.nodes():
#     list_of_smiles=[]
#     gr
#     list_of_smiles.append(atom.element)
#     for e in G.edges():
#         list_of_smiles.append(e.weight)
#
#     for neighbour in G.neighbors(atom):
#         list_of_smiles.append(neighbor.element, neighbor.atom_number) neighbor.element == "H":
#                 alcohols.append([atom, neighbour])
#                 # print(f"Alcohol: {str(atom), str(neighbour)}")
#             else:
#                 pass
#
# nx.nodes()
# def finding_smiles(G, cycle):
#     list_of_nodes=[]
#     for i, atom in enumerate(cycle):
#         list_of_nodes.append(atom)
#
# def edge_weights(G,cycle):
#     for e in G.edges().data(nbunch=N1):
#         if e[1] == N2: 3
#             return e[2]["weight"]
#


def list_comparison(list_of_lists, list_2):

    for element in list_of_lists:
        if element == list_2:
            return True

    return False

def same_cycle(list_of_lists, list_2):
    if list_comparison() is True:

        pass

if __name__ == "__main__":

    G1 = nx.OrderedGraph()
    Nodes_1 = [Atom("C", n) for n in range(12)]

    G1.add_nodes_from(Nodes_1)
    Nodes_1.append(Nodes_1[0])

    G1.add_path(Nodes_1, weight=3)
    G1.add_path([Nodes_1[0], Nodes_1[5]], weight=1)
    cycle = Nodes_1[:6]

    for c in smiles(G1, cycle):
        
        print("".join(c))

