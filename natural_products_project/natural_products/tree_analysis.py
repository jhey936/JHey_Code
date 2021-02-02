import collections
from natural_products.isomorphism import check_isomorphism

import networkx as nx


def add_element(tree, semantic_analysis, cycle, ring_counter, recursion_depth=0):
     # # print("ADDING CYCLE", [a.element for a in cycle.nodes])
    try:
        existing_cycle, count = get_tree_value(tree, semantic_analysis)
        # # print("EXISTING cycle", existing_cycle, [a.element for a in existing_cycle.nodes], count)

        if isinstance(existing_cycle, nx.Graph):
            # # print("Checking isomorphism", existing_cycle)
            isomorphic = check_isomorphism(cycle, existing_cycle)  # make this increment a counter

            if isomorphic:
                # # print(f"Cycle: {cycle} in graph")
                ring_counter(semantic_analysis)

            else:
                print("FML", [a.element for a in existing_cycle.nodes])
                # raise AttributeError

    except ValueError:

        if len(semantic_analysis) == 1:
            # # print("Here", [cycle, 0])

            if isinstance(semantic_analysis[0], list):

                path = ",".join([str(i) for i in semantic_analysis[0]])
                
                tree[path] = [cycle, 0]
            else:
                tree[semantic_analysis[0]] = [cycle, 0]

        else:
            recursion_depth+=1
            add_element(tree[semantic_analysis[0]], semantic_analysis[1:], cycle, ring_counter, recursion_depth)

    if recursion_depth == 0:
        ring_counter(semantic_analysis)


def get_tree_value(tree, path):
    if len(path) == 1:
        if isinstance(path[0], list):
            # print(path[0])
            return tree[",".join([str(i) for i in path[0]])]
        return tree[path[0]]
    else:
        return get_tree_value(tree[path[0]], path[1:])


def tree_creation():
    tree = lambda: collections.defaultdict(tree)
    cycle_tree = tree()
    return cycle_tree

