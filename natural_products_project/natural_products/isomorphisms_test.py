import networkx as nx
import networkx.algorithms.isomorphism as iso
from natural_products.io_utils import Atom



def check_isomorphism(G1, G2):
    """Check if two graphs are isomorphic"""
    em = iso.categorical_edge_match("weight", None)
    nm = iso.categorical_node_match("element", None)

    return nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em)


if __name__ == "__main__":

    G1 = nx.OrderedGraph()
    G2 = nx.OrderedGraph()
    Nodes_1 = [Atom("C", n) for n in range(6)]
    Nodes_2 = [Atom("C", n) for n in range(6)]
    G1.add_nodes_from(Nodes_1)
    G2.add_nodes_from(Nodes_2)
    # G1.add_node(5, **{"fill": "blue"})

    # G2.add_nodes_from(["a", "b", "c", "d", "e", "f"])
    # G2.add_node(50, **{"fill": "blue"})
    Nodes_1.append(Nodes_1[0])
    G1.add_path(Nodes_1, weight=1)
    Nodes_2.append(Nodes_2[0])
    G2.add_path(Nodes_2)
    G1.add_edge(Nodes_1[1], Nodes_1[2], weight=3)
    G1.add_edge(Nodes_1[3], Nodes_1[4], weight=5)

    nm = iso.categorical_node_match('element', None)
    em = iso.categorical_edge_match('weight', None)
    # print(nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em))
    # print([i for i in nx.generate_edgelist(G1, data=["weight"])])


    #
    # G1 = nx.OrderedGraph()
    # G2 = nx.OrderedGraph()
    # Nodes_1 = [Atom("C", n) for n in range(6)]
    # Nodes_2 = [Atom("C", n) for n in range(6)]
    # G1.add_nodes_from(Nodes_1)
    # G2.add_nodes_from(Nodes_2)
    # # G1.add_node(5, **{"fill": "blue"})
    #
    # # G2.add_nodes_from(["a", "b", "c", "d", "e", "f"])
    # # G2.add_node(50, **{"fill": "blue"})
    # Nodes_1.append(Nodes_1[0])
    # G1.add_path(Nodes_1, weight=1)
    # Nodes_2.append(Nodes_2[0])
    # G2.add_path(Nodes_2)
    # G1.add_edge(Nodes_1[1], Nodes_1[2], weight=3)
    # G1.add_edge(Nodes_1[3], Nodes_1[4], weight=5)
    #
    # nm = iso.categorical_node_match('element', None)
    # em = iso.categorical_edge_match('weight', None)
    # # print(nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em))
    # # print([i for i in nx.generate_edgelist(G1, data=["weight"])])
#
# , fill='red'
# cycle = ["a", "b", "c", "d"]
#     G = nx.Graph()
#
#     G.add_nodes_from(cycle)
#
#     G.add_weighted_edges_from([("C", "C", 1), ("C", "O", 2), ("O", "N", 3), ("N", "S", 4), ("S", ) ])
#
#     edge_weights = get_cycle_edge_weights(G, cycle)
#
#     # print(edge_weights)  # should # print [1,2,3,4]
#
# # print(Count_atoms(Graph, allcycles))