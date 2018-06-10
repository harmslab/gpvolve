#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# Input: Dictionary of paths (tuple of genotype strings) with respective probability.
# Example input: {('00','10','11'): 0.8, ('00','10','11'): 0.2}

# Visualize the output of any path sampling method as a flat genotype-phenotype network graph with nodes representing
# genoypes and edges representing paths between nodes.
# Also load gpmap of the dataset

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import json
import sys
from operator import mul
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from gpgraph import base
from gpmap import GenotypePhenotypeMap
from evotpt.sampling import utils


class GenotypePhenotypeGraph(object):
    def __init__(self, gpm=None, mutant=None):
        # Read genotype-phenotype map
        self.read_gpm(gpm)

        # Set filename.
        self.outfilename = sys.argv[1].split(".")[0].split("/")[-1]

        # Set mutant. Default: The furthest genotypes from wildtype by hamming-distance.
        if mutant == None:
            self.mutant = utils.furthest_genotypes(self.data, self.wildtype, self.data.genotypes)[0]
        else:
            self.mutant = mutant

        # Define trajectory length.
        self.traject_length = utils.signed_hamming_distance(self.wildtype, self.wildtype, self.mutant)

    def read_gpm(self, gpm):
        """Assign GenotypePhenotypeMaps properties to variables"""

        # Add gpm.
        self.gpm = gpm
        self.data = self.gpm.data

        # Set wildtype.
        self.wildtype = gpm._wildtype

        # Set mutaions.
        self.mutations = gpm.mutations

        # Get binary of wildtype:
        # First: get index of wildtype in the dataframe.
        genotype_index = self.data.index[self.data['genotypes'] == self.wildtype].tolist()
        # Second: Pull out the corresponding binary from the dataframe and set as binary wildtype.
        self.binary_wildtype = self.data.iloc[genotype_index[0]]['binary']


    def define_xy_coordinates(self):
        y = 0
        x = 0
        node_coordinates = {self.wildtype: [x, y]}
        nodelist = [self.wildtype]

        # Iterate over levels of genotypes. Each level corresponds to one additional mutation.
        for step in range(0, self.traject_length):
            temp_node_list = []
            y += 1
            neighborlist = []
            # Get all the genotypes that have exactly one more mutation than the current genotypes.
            for node in nodelist:
                neighborlist = neighborlist + list(utils.get_neighbors(self.data, self.wildtype, node, self.mutations, reversibility=False))
            x = 0
            # Get unique list of neighbors and set coordinates for each.

            for neighbor in list(set(neighborlist)):
                node_coordinates[neighbor] = [x, y]
                temp_node_list.append(neighbor)
                x += 1
            # Align nodes to center/wildtype.
            for node_ in temp_node_list:
                print(node_, len(temp_node_list))
                node_coordinates[node_][0] = node_coordinates[node_][0] - ((len(temp_node_list)-1) / 2)
            # Update nodelist
            nodelist = temp_node_list
        return node_coordinates

    def draw_map(self):
        for node, coordinates in self.define_xy_coordinates().items():
            plt.scatter(*coordinates)

        # Invert y-axis
        plt.gca().invert_yaxis()
        plt.savefig("%s_map.pdf" % self.outfilename, format='pdf', dpi=300)

gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
gpraph = GenotypePhenotypeGraph(gpm)

gpraph.draw_map()