#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# rearrange branch

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import json
import sys
import random
from math import e
from scipy.stats import rv_discrete
from operator import mul
from functools import reduce
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt import utils
from gpmap import GenotypePhenotypeMap
from gpmap.utils import hamming_distance


class Sampling(object):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm=None, mutant=None):
        self.read_gpm(gpm)


        # Set mutant. Default: The furthest genotypes from wildtype by hamming-distance.
        if mutant == None:
            self.mutant = utils.furthest_genotypes(self.data, self.wildtype, self.data.genotypes)[0]
        else:
            self.mutant = mutant

        print("Mutant genotype(s) set to: %s. Mutant genotype(s) defines the (desired) end of each path. \n[Default: Furthest genotype(s) from wildtype.]" % self.mutant)

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

    def sample(self, iterations, random_seed):
        """Sample paths by iterating self.simulate. Save output as .json file"""

        # Create masterlog dictionary, a list for all paths sampled and add random seed to dictionary
        masterlog = {}
        random.seed(random_seed)
        masterlog["random_seed"] = random_seed
        paths = []

        # Lag intervals saves the steps at which convergence measurement was taken; Lag k defines interval size;
        # Previous dictionary resembles the pmf at step 0; Convergence saves the euclidean distances between pmf at
        # each lag interval.
        lag_intervals = [100]
        lag_k = 100
        prev_dict = {}
        convergence = []

        # Iterate self.sample and add every resulting path to masterlog.
        # self.simulate is defined in the descending Class, e.g. mcmc_path_sampling.py
        for i in range(1, int(iterations)+1):
            log = self.simulate()
            masterlog[str(i)] = log

            paths.append(tuple(log["path"]))

            if i == lag_intervals[-1]:
                print("Path %s: %s" % (i, masterlog[str(i)]["path"]))
                # Once the last interval is reached, add the next interval, i.e. last interval + lag k.
                lag_intervals.append(lag_intervals[-1] + lag_k)

                # Get the pmf after i amount of steps as.
                curr_dict = self.pmf(paths)

                euclid_dist = 0
                # Calc. euclidean distance of each path in current pmf to pmf at step current - k (current - previous).
                for path in curr_dict:
                    try:
                        euclid_dist += abs(curr_dict[path] - prev_dict[path])
                    # If new path has been sampled since current step - k, euclidean distance equals prob. of that path.
                    except KeyError:
                        euclid_dist += curr_dict[path]

                convergence.append(euclid_dist)
                # Set current pmf as previous pmf.
                prev_dict = curr_dict
        # Remove last interval because that won't be reached.
        lag_intervals.pop(-1)

        plt.plot(lag_intervals, convergence)
        # plt.hlines(0, xmin=-1000, xmax=lag_intervals[-1]**2, color='black', linestyle=':', linewidth= 2)
        plt.savefig("%s_convergence.pdf" % self.outfilename, format='pdf', dpi=300)


        # Output masterlog as a .json file.
        with open("%s.json" % self.outfilename, 'w') as outfile:
            json.dump(masterlog, outfile, sort_keys=True, indent=4)

        return masterlog


    def pmf(self, paths):
        dict = {}
        counts, paths = self.count(paths)
        counts_sum = sum(counts)
        for path, count in zip(paths, counts):
            dict[path] = count/counts_sum
        # pmf = [count/counts_sum for count in counts]
        return dict

    def count(self, paths):
        counts = []
        uniq_paths = self.unique_sorted_paths(paths)
        for path in uniq_paths:
            counts.append(paths.count(path))
        return tuple(counts), uniq_paths

    def unique_sorted_paths(self, paths):
        """ Return unique sorted list of path IDs """
        uniq = list(set(paths))
        uniq_sorted_pathlist = sorted(uniq, key=lambda tup: tup[:])
        return uniq_sorted_pathlist