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
        masterlog["paths"] = []

        # Lag intervals saves the steps at which convergence measurement was taken; Lag k defines interval size;
        # Previous dictionary resembles the pmf at step 0; Convergence saves the euclidean distances between pmf at
        # each lag interval.
        lag_intervals = [100]
        lag_k = 100
        prev_path_counts = {}
        convergence = []

        # Iterate self.sample and add every resulting path to masterlog.
        # self.simulate is defined in the descending Class, e.g. mcmc_path_sampling.py
        for i in range(1, int(iterations)+1):
            print("Path %s" %i)

            log = self.simulate()
            masterlog['paths'].append(tuple(log["path"]))

            # CONVERGENCE
            if i == lag_intervals[-1]:
                # Once the latest interval is reached, add the next interval, i.e. last interval + lag k.
                lag_intervals.append(lag_intervals[-1] + lag_k)
                # Create empty dictionary for current path counts.
                curr_path_counts = {}
                # Get the path counts since the last counting, i.e last lag_interval.
                path_counts = self.count(masterlog['paths'][-lag_k:])

                # Add the new counts to the previous counts to get the current path counts.
                for path, count in path_counts.items():
                    try:
                        curr_path_counts[path] = prev_path_counts[path] + count
                    # If a new path has been sampled since the previous counting, add the new path to the dictionary.
                    except KeyError:
                        curr_path_counts[path] = count

                # CONVERGENCE CRITERION: Euclidean distance between current probability mass function (pmf) and the
                # previous pmf.
                convergence.append(self.euclidean_distance(self.pmf(prev_path_counts, i-lag_k), self.pmf(curr_path_counts, i)))

                # Set the current path counts as previous path counts before starting the next iteration.
                prev_path_counts = curr_path_counts

        # Remove last interval because that won't be reached.
        lag_intervals.pop(-1)
        # Plot convergence and save as .pdf file in current working directiory.
        plt.plot(lag_intervals, convergence)
        plt.savefig("%s_convergence.pdf" % self.outfilename, format='pdf', dpi=300)

        # Add probability mass function of all sampled paths to masterlog.
        pmf = {}
        for path, prob in self.pmf(self.count(masterlog['paths']), len(masterlog['paths'])).items():
            pmf[",".join(path)] = prob
        masterlog['pmf'] = pmf

        # Output masterlog as a .json file.
        with open("%s.json" % self.outfilename, 'w') as outfile:
            json.dump(masterlog, outfile, sort_keys=True, indent=4)

        return masterlog

    def euclidean_distance(self, prev_pmf, current_pmf):
        euclid_dist = 0
        # Calc. euclidean distance of each path in current pmf to pmf at step current - k (current - previous).
        for path in current_pmf:
            try:
                euclid_dist += abs(current_pmf[path] - prev_pmf[path])
            # If a new path has been sampled since last lag interval, euclidean distance = prob. of that path.
            except KeyError:
                euclid_dist += current_pmf[path]
        return euclid_dist

    def pmf(self, path_counts, total_count):
        dict = {}
        try:
            for path, count in path_counts.items():
                dict[path] = count/total_count
        except ZeroDivisionError:
                dict[path] = 0
        return dict

    def count(self, paths):
        dict = {}
        uniq_paths = utils.unique_sorted_paths(paths)
        for path in uniq_paths:
            dict[path] = paths.count(path)
        return dict