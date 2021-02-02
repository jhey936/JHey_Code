



def count_atoms(G, cycles):
    number_of_atoms=[]
    for c in cycles:
        number_of_atoms.append(len(c))
    return number_of_atoms

# print(count_atoms(myGraph, all_cycles))