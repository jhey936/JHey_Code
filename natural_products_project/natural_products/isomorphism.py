import networkx as nx
import networkx.algorithms.isomorphism as iso
# from natural_products.io_utils import *

#
# def check_isomorphism(G1, G2):
#     """Check if two graphs are isomorphic"""
#     em = iso.categorical_edge_match("weight", None)
#     nm = iso.categorical_node_match("element", None)
#
#     return nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em)


def _semantic_feasibility(self, G1_node, G2_node):
    """Returns True if mapping G1_node to G2_node is semantically feasible.
    """
    # Make sure the nodes match
    if self.node_match is not None:

        nm = self.node_match(G1_node, G2_node)
        if not nm:
            return False

    # Make sure the edges match
    if self.edge_match is not None:

        # Cached lookups
        G1_adj = self.G1_adj
        G2_adj = self.G2_adj
        core_1 = self.core_1
        edge_match = self.edge_match

        for neighbor in G1_adj[G1_node]:
            # G1_node is not in core_1, so we must handle R_self separately
            if neighbor == G1_node:
                if not edge_match(G1_adj[G1_node][G1_node],
                                  G2_adj[G2_node][G2_node]):
                    return False
            elif neighbor in core_1:
                if not edge_match(G1_adj[G1_node][neighbor],
                                  G2_adj[G2_node][core_1[neighbor]]):
                    return False
        # syntactic check has already verified that neighbors are symmetric

    return True


class CustomGraphMatcher(nx.algorithms.isomorphism.GraphMatcher):
    semantic_feasibility = _semantic_feasibility


def  get_IsomorphismChecker(G1, G2):

    nm = lambda x, y: x.element == y.element

    em = iso.categorical_edge_match('weight', None)

    return CustomGraphMatcher(G1, G2, node_match=nm, edge_match=em)


def check_isomorphism(G1, G2):
    checker = get_IsomorphismChecker(G1, G2)

    return checker.is_isomorphic()


if __name__ == "__main__":

    G1 = nx.Graph()
    G2 = nx.Graph()
    Nodes_1 = [Atom("N", 1), Atom("O", 2), Atom("C", 3), Atom("C", 4), Atom("C", 5), Atom("C", 6)]
    Nodes_2 = [Atom("C", 1), Atom("N", 2), Atom("C", 3), Atom("O", 4), Atom("C", 5), Atom("C", 6)]

    G1.add_nodes_from(Nodes_1)
    G2.add_nodes_from(Nodes_2)

    # print([a.element for a in G1.nodes])
    # print([a.element for a in G2.nodes])

    Nodes_1.append(Nodes_1[0])
    G1.add_path(Nodes_1, weight=[1])
    G1.add_edge(Nodes_1[1], Nodes_1[2], weight=1)
    G1.add_edge(Nodes_1[3], Nodes_1[4], weight=1)
    Nodes_2.append(Nodes_2[0])
    G2.add_path(Nodes_2, weight=[1])
    G2.add_edge()

    isomorphism_checker = get_IsomorphismChecker(G1, G2)

    # print(isomorphism_checker.is_isomorphic())

    # G1 = nx.Graph()
    # G2 = nx.Graph()
    # Nodes_1 = [Atom("N", 1), Atom("C", 2), Atom("C", 3), Atom("C", 4), Atom("C", 5), Atom("C", 6)]
    # Nodes_2 = [Atom("C", 1), Atom("N", 2), Atom("C", 3), Atom("C", 4), Atom("C", 5), Atom("C", 6)]
    #
    # G1.add_nodes_from(Nodes_1)
    # G2.add_nodes_from(Nodes_2)
    #
    # # print([a.element for a in G1.nodes])
    # # print([a.element for a in G2.nodes])
    #
    # Nodes_1.append(Nodes_1[0])
    # G1.add_path(Nodes_1, weight=[1])
    # G1.add_edge(Nodes_1[1], Nodes_1[2], weight=3)
    # G1.add_edge(Nodes_1[3], Nodes_1[4], weight=1)
    # Nodes_2.append(Nodes_2[0])
    # G2.add_path(Nodes_2, weight=[1])
    # G2.add_edge()
    #
    # isomorphism_checker = get_IsomorphismChecker(G1, G2)
    #
    # # print(isomorphism_checker.is_isomorphic())
