#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# Visualize the output of any path sampling method
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

from gpmap import GenotypePhenotypeMap
from evotpt.sampling import Sampling


class GenotypePhenotypeGraph(object):
    def __init__(self, gpm=None, mutant=None):
        self.read_gpm(gpm)

        print(self.data)

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

    # def x_coordinates(self):
    #     for genotype in self.data.binary








gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
gpraph = GenotypePhenotypeGraph(gpm)