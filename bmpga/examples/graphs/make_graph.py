# coding=utf8
"""

bmpga: A program for finding global minima
Copyright (C) 2018- ; John Hey
This file is part of bmpga.

bmpga is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License V3.0 as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

bmpga is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License V3.0 for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


=========================================================================
John Hey (Created: 5-11-18)

Utility to create and analyse molecular graphs

=========================================================================
"""


import networkx as nx

from glob import glob

from bmpga.utils.io_utils import XYZReader

reader = XYZReader()


XYZs = glob("test_data/*.xyz")

print(XYZs)


for xyz in XYZs:
    print(xyz)
    cluster = reader.read(xyz, return_clusters=True)[0]
    print(cluster.molecules)

    graph = cluster.molecules[0].to_graph()

    nodes = graph.nodes

    for node in nodes:
        print(node.__dict__)

    print(graph.nodes)


    print(graph.edges)

    cycles = nx.cycle_basis(graph)

    print(cycles)
    print(len(cycles))

    print(nx.minimum_cycle_basis(graph))

