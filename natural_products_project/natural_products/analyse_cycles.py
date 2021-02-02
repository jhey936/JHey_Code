import networkx as nx
import unittest
import itertools
import matplotlib.pyplot as plt
import copy


class GraphCycleAnalyser:
    def __init__(self, G, verbose=True):

        self.G = G
        self.verbose = verbose

    # @staticmethod
    # def true_cycle(G):
    #     for node in G.nodes:
    #         if nx.degree(G,node)==1:
    #             return False
    #     for edge in G.edges:
    #         G2 = nx.Graph(G.subgraph(G.nodes))
    #         G2.remove_edge(edge[0], edge[1])
    #         if not nx.is_connected(G2):
    #             return False
    #     return True

    # @staticmethod
    # def symmetric_difference_or_none(A, B):

    #     sym_diff = A.symmetric_difference(B)

    #     if len(sym_diff) == len(A)+len(B):
    #         return {}
    #     else:
    #         return sym_diff

    @staticmethod
    def strip_leaves(G):
        """Returns a copy of G with leaf nodes stripped out."""
        G = copy.deepcopy(G)
        modified = True

        while modified:

            modified = False
            # print(G)
            temp_G = copy.deepcopy(G)

            for node, degree in temp_G.degree():

                # print("Test")
                if degree == 1:
                
                    modified = True
                    G.remove_node(node)

        return G

    @staticmethod
    def dfs(graph, start, end):
        fringe = [(start, [])]
        while fringe:
            state, path = fringe.pop()
            if path and (state == end) and (len(path) >2):
                yield path
                continue
            for next_state in graph[state]:
                if next_state in path:
                    continue
                fringe.append((next_state, path+[next_state]))

    def find_all_cycles(self):

        self.G = GraphCycleAnalyser.strip_leaves(self.G)

        G_dict = nx.to_dict_of_lists(self.G)

        cycles = [path  for node in G_dict for path in GraphCycleAnalyser.dfs(G_dict, node, node)]
        cycles = [path for path in cycles if len(path) > 2]
        cycles = set([tuple(sorted(cycle, key= lambda x: x.atom_number)) for cycle in cycles])
        cycles = [nx.Graph(self.G.subgraph(cycle)) for cycle in cycles]
        return cycles


class GraphCycleAnalyser_OLD:
    def __init__(self, G, verbose=True):
        self.G = G
        self.verbose = verbose

    def find_all_cycles(self):

        cycle_basis = nx.cycle_basis(self.G)

        cycle_graphs = [nx.Graph(self.G.subgraph(c)) for c in cycle_basis]

        edge_sets = [set([e for e in graph.edges]) for graph in cycle_graphs]

        new_edge_sets = self.analyse_cycles(edge_sets)

        new_node_sets = []
        for es in new_edge_sets:

            node_set = []

            for edge in es:
                node_set.append(edge[0])
                node_set.append(edge[1])

            if node_set not in new_node_sets:
                new_node_sets.append(tuple(set(node_set)))

        new_node_sets = set([ns for ns in new_node_sets])
        cycle_graphs = [nx.Graph(self.G.subgraph(c)) for c in new_node_sets]

        cycle_graphs = [g for g in cycle_graphs if self.true_cycle(g)]

        return cycle_graphs

    @staticmethod
    def true_cycle(G):
        for node in G.nodes:
            if nx.degree(G,node)==1:
                return False
        for edge in G.edges:
            G2 = nx.Graph(G.subgraph(G.nodes))
            G2.remove_edge(edge[0], edge[1])
            if not nx.is_connected(G2):
                return False
        return True

    @staticmethod
    def symmetric_difference_or_none(A, B):

        sym_diff = A.symmetric_difference(B)

        if len(sym_diff) == len(A)+len(B):
            return {}
        else:
            return sym_diff

    def analyse_cycles(self, edge_sets):

        update = False

        new_edge_sets = []

        for A, B in itertools.combinations(edge_sets, 2):

            new_cycle = self.symmetric_difference_or_none(A, B)

            if len(new_cycle) != 0 and new_cycle not in edge_sets:

                update = True

                new_edge_sets.append(new_cycle)

        new_edge_sets.extend(edge_sets)

        if update:
            return self.analyse_cycles(new_edge_sets)
        else:
            return new_edge_sets

def plot_labelled_graph(G):

    nx.draw(G, labels=dict(zip(G.nodes, [str(n) for n in G.nodes])))
    plt.show()

class TestGraphAnalysis(unittest.TestCase):

    def test_static_diff(self):

        s1 = {1,2,3,4}
        s2 = {4,5,6,7}
        s3 = {5,6,7,8}

        sd = GraphCycleAnalyser.symmetric_difference_or_none


        rs1 = sd(s1, s2)
        rs2 = sd(s1, s3)

        self.assertEqual(len(rs1), 6)

        self.assertEqual(len(rs2), 0)

    def test_napthalene(self):
        """construct & test napthalene-like fused rings"""
        G = nx.Graph()
        G.add_nodes_from(range(1,11))
        G.add_path((1,2,3,4,5,6,1))
        G.add_path((5,6,7,8,9,10,5))
        cycles = self.get_cycles(G)

        self.assertEqual(len(cycles), 3)

    def test_octamer_bridge(self):
        # Construct an octameric ring with a bridge import TestCase
        G = nx.Graph()
        G.add_nodes_from(range(1,10))
        G.add_path((1,2,3,4,5,6,7,8,1))
        G.add_path((3,9,7))

        cycles = self.get_cycles(G)
        self.assertEqual(len(cycles), 3)

    def test_edge_sharing_squares(self):
        # Construct three edge-sharing squares
        G = nx.Graph()
        G.add_nodes_from(range(1,9))
        G.add_path((1,2,3,4,8))
        G.add_path((1,5,6,7,8))
        G.add_edges_from([(2,6),(3,7)])

        cycles = self.get_cycles(G)
        with open("tst", "w") as of:
            of.write("sqs"+str(cycles))
        self.assertEqual(len(cycles), 6)

    def test_kite(self):
        # Construct a kite
        G = nx.Graph()
        G.add_nodes_from(range(1, 6))
        G.add_path((1, 4, 3, 5, 1))
        G.add_path((1, 2, 3))
        G.add_path((4, 2, 5))

        cycles = self.get_cycles(G)

        self.assertEqual(len(cycles), 10)

    def test_corner_sharing_trimers(self):
        # Construct corner-sharing trimers
        G = nx.Graph()
        G.add_nodes_from(range(1,6))
        G.add_path((1,2,3,1))
        G.add_path((1,4,5,1))

        cycles = self.get_cycles(G)
        self.assertEqual(len(cycles), 2)

    def test_steroid_rings(self):
        # Construct the basic steroid ring system
        G = nx.Graph()
        G.add_nodes_from(range(1,18))
        G.add_path((1,2,3,4,5,6,1))
        G.add_path((3,4,10,9,8,7,3))
        G.add_path((7,8,14,13,12,11,7))
        G.add_path((13,14,17,16,15,13))

        cycles = self.get_cycles(G)

        self.assertEqual(len(cycles), 10)

    def get_cycles(self, G):

        analyser = GraphCycleAnalyser(G, verbose=False)
        all_cycles = analyser.find_all_cycles()
        return all_cycles

if __name__ == "__main__":

    G = nx.Graph()

    G.add_path([1,2,3,4,5,6])

    G.add_path([1,2,3,4,5,6,1])
    # G.add_path([1,2,3,1])

    SL = GraphCycleAnalyser.strip_leaves

    # print(SL(G).nodes)

    #get_dict = GraphCycleAnalyser2.create_dict_no_small_loops
    
    find_all_cycles = GraphCycleAnalyser.find_all_cycles
    # print(find_all_cycles(G))
