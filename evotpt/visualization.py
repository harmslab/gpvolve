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
from matplotlib import cm

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from gpgraph import base
from gpmap import GenotypePhenotypeMap
from evotpt.sampling import utils


class GenotypePhenotypeGraph(object):
    def __init__(self, gpm, sampling_results=None, mutant=None):
        # Read genotype-phenotype map
        self.read_gpm(gpm)

        # Read sampling results of respective map
        self.pmf = self.read_sampling_results(sampling_results)

        self.fluxes = self.flux(self.pmf)
        print(self.fluxes)
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

    def read_sampling_results(self, jsonfile):
        f = open(jsonfile, "r")
        data = json.load(f)
        pmf = data['pmf']

        return pmf


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

            for neighbor in sorted(list(set(neighborlist))):
                node_coordinates[neighbor] = [x, y]
                temp_node_list.append(neighbor)
                x += 1
            # Align nodes to center/wildtype.
            for node_ in temp_node_list:
                node_coordinates[node_][0] = node_coordinates[node_][0] - ((len(temp_node_list)-1) / 2)
            # Update nodelist
            nodelist = temp_node_list
        return node_coordinates

    def draw_map(self):
        # Get dicitonary with color for each node and choose color map
        colors = self.colors()
        cmap = plt.cm.get_cmap('plasma')

        node_coordinates = self.define_xy_coordinates()
        for node, coordinates in node_coordinates.items():
            neighbors = list(utils.get_neighbors(self.data, self.wildtype, node, self.mutations, reversibility=False))
            for neighbor in neighbors:
                linewidth = 0
                if tuple([node, neighbor]) in self.fluxes:
                    linewidth = self.fluxes[tuple([node,neighbor])]
                # print(coordinates, node_coordinates[neighbor])
                # x_y = list(zip(coordinates, node_coordinates[neighbor]))
                x = list(zip(coordinates, node_coordinates[neighbor]))[0]
                y = list(zip(coordinates, node_coordinates[neighbor]))[1]
                # Draw lines between neighbors
                plt.plot(x, y, '-', color='grey', linewidth=linewidth*8, zorder=0)

            # Get color for node from colors dictionary.
            color = cmap(colors[node])
            # Draw nodes.
            plt.scatter(*coordinates, color=color, zorder=1, s=300)
            # Add genotype labels for node.
            plt.annotate(node, coordinates, size=5, ha='center', va='center', zorder=2)



        # Invert y-axis
        plt.gca().invert_yaxis()
        plt.savefig("%s_map.pdf" % self.outfilename, format='pdf', dpi=300)

    def colors(self):
        # Define a color value between 0 and 1 for each phenotype,
        # where the smallest and largest phenotype are 0 and 1, respectively.

        phenotypes = {}
        phenotype_list = []

        # Get all phenotypes
        for genotype in self.data.genotypes:
            pheno = utils.get_phenotype(self.data, genotype)
            phenotype_list.append(pheno)
            phenotypes[genotype] = pheno

        # Define smallest phenotype and largest phenotypes.
        min_phe = min(phenotype_list)
        max_phe = max(phenotype_list)

        color_values = {}
        for genotype, phen in phenotypes.items():
            # By substracting smallest phenotype and dividing by largest genotype(-smallest phenotype),
            # all phenotypes are mapped onto a scale of 0 to 1.
            color_values[genotype] = (phen - min_phe) / (max_phe - min_phe)

        return color_values

    def flux(self, pmf):
        dict = {}
        for path_str, prob in pmf.items():
            path = tuple(path_str.split(","))
            for i in range(1, len(path)):
                step = (path[i-1],path[i])
                try:
                    dict[step] += prob
                except KeyError:
                    dict[step] = prob
        return dict


gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
gpraph = GenotypePhenotypeGraph(gpm, sys.argv[2])

gpraph.draw_map()