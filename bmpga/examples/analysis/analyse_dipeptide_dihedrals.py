"""

Call with:
python analyse_dipeptide_dihedrals.py <regex to lowest files> <path to output>

eg:
python analyse_dipeptide_dihedrals.py NALA_CALA/water_5/*/GRlowest NALA_CALA_5_wat_dihedrals.csv

"""
import os
import sys
import glob
import pandas as pd

from bmpga.utils.io_utils import XYZReader
from bmpga.analysis.peptides import Peptide

reader = XYZReader()

files = glob.glob(sys.argv[1]+"GRlowest*")
outfile = os.path.abspath(sys.argv[2])
print(f"Reading files: {files}, writing to {outfile}")

energies = []
dihedrals = []

clusters = []

for file_name in files:
    clusters.extend(reader.read(os.path.abspath(file_name)))
    print(f"found {len(clusters)} in {os.path.abspath(file_name)}")


for cluster in clusters:
    peptide = Peptide(coordinates=cluster.molecules[0].coordinates,
                      particle_names=cluster.molecules[0].particle_names)

    energies.append(cluster.cost)
    diheds = peptide.calculate_dihedrals()
    for key in diheds.keys():
        diheds[key] = diheds[key][0]
    dihedrals.append(diheds)  # [(key, value) for key, value in zip(diheds.keys(), diheds.values())])


dihedrals = pd.DataFrame(data=dihedrals, columns=["phi", "psi", "w"], index=energies)

dihedrals.sort_index()

dihedrals.to_csv(outfile)

print(dihedrals)
