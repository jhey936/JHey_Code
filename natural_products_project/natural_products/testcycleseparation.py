import networkx as nx

if __name__ == "__main__":
    cycle = ["a", "b", "c", "d"]

    G = nx.Graph()

    G.add_nodes_from(cycle)

    G.add_weighted_edges_from([("C", "C", 1), ("C", "O", 2), ("O", "N", 3), ("N", "S", 4), ("S", ) ])

    edge_weights = get_cycle_edge_weights(G, cycle)

    # print(edge_weights)  # should # print [1,2,3,4]

# print(Count_atoms(Graph, allcycles))