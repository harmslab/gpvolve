#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# -------------------------------------------------------------------------
# USAGE: mcmc_path_sampling.py <dataset>.json <outputfile>.json
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import pandas as pd
import sys
from scipy.stats import rv_discrete
from operator import mul
from functools import reduce
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt.sampling import Sampling
from gpmap import GenotypePhenotypeMap

class ExhaustiveSampling(Sampling):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm, outputfile, population_size, null_steps=False, **kwargs):
        Sampling.__init__(self, gpm)

        # Set filename.
        self._filename = outputfile

        # Set population size.
        self.pop_size = population_size

        # Set reversibility
        self.reversibility = False

        # Set null_steps
        self.null_steps = null_steps

        # Define trajectory length.
        self.traject_length = self.signed_hamming_distance(self.wildtype, self.mutant)

        self.tm = self.transition_matrix()

    def plot(self):
        fig, ax = plt.subplots()
        y = self.possible_paths()
        x = [i for i in range(1, len(y)+1)]

        plt.bar(x,y)

        plt.savefig(sys.argv[2], format='pdf', dpi=300)

    def possible_paths(self):
        nonzero_probs = []
        nonzero_paths = []
        probs, paths = self.path_probabilities()
        for prob, path in zip(probs, paths):
            if prob > 0:
                nonzero_probs.append(prob)
                nonzero_paths.append(path)

        nonzero_paths, nonzero_probs = zip(*sorted(zip(nonzero_paths, nonzero_probs)))
        print(nonzero_paths, nonzero_probs)
        return tuple(nonzero_probs)

    def all_paths(self):
        paths = [[self.binary_wildtype]]
        temp_path_memory = []

        for step in range(0, self.traject_length):
            for path in paths:
                for neighbor in self.get_neighbors(path[-1]):
                    temp_path_memory.append(path + [neighbor])
            paths = temp_path_memory
            temp_path_memory = []

        paths_tup = [tuple(path) for path in paths]

        return paths_tup

    def path_probabilities(self):
        path_probs = []
        pathlist = []
        for path in self.all_paths():
            pathlist.append(path)
            transition_probs = []
            for genotype in path[1:]:
                transition_prob = self.tm.at[path[path.index(genotype)-1], genotype]
                transition_probs.append(transition_prob)
            path_probs.append(transition_probs)

        path_probs_products = []
        for path_prob in path_probs:
            path_probs_products.append(reduce(mul, path_prob, 1))

        return(path_probs_products, pathlist)

if __name__ == "__main__":
    # execute only if run as a script
    gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
    exhaustive_sampling = ExhaustiveSampling(gpm, outputfile=sys.argv[2], population_size=10000, null_steps=False)
    exhaustive_sampling.plot()
