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

from evoTPT.sampling import Sampling
from gpmap import GenotypePhenotypeMap

class ExhaustiveSampling(Sampling):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm, outputfile, population_size, reversibility=True, **kwargs):
        Sampling.__init__(self, gpm)

        # Set filename.
        self._filename = outputfile

        # Set population size.
        self.pop_size = population_size

        # Set reversibility
        self.reversibility = reversibility

        print("Reverse steps allowed: %s. Default: True" % reversibility)

    def all_paths(self):
        path1, path2, path3, path4, path5, path6 = [],[],[],[],[],[]
        path1 = [self.binary_wildtype]
        for gt1 in self.get_neighbors(path1[-1]):

            path2.append(path1+[gt1])
        # print(path2)

        for pa2 in path2:
            for gt2 in self.get_neighbors(pa2[-1]):
                path3.append(pa2+[gt2])

        for pa3 in path3:
            for gt3 in self.get_neighbors(pa3[-1]):
                path4.append(pa3+[gt3])

        for pa4 in path4:
            for gt4 in self.get_neighbors(pa4[-1]):
                path5.append(pa4+[gt4])

        for pa5 in path5:
            for gt5 in self.get_neighbors(pa5[-1]):
                path6.append(pa5+[gt5])

        path_ids = []
        for path in path6:
            path_id = []
            for genotype in path:
                # Get index of genotype.
                genotype_id = self.data.index[self.data['genotypes'] == genotype].tolist()
                path_id.append(genotype_id[0])
            # make path ID a tuple and add to masterlog.
            path_id = tuple(path_id)
            path_ids.append(path_id)

        pathprobs = []
        for path in path6:
            prob = []
            for i in range(0,len(path)-1):
                prob.append(max(0, self.fixation_probability(self.get_phenotype(path[i]), self.get_phenotype(path[i+1]), self.pop_size)))
            pathprobs.append(reduce(mul, prob, 1))

        paths_sum = sum(pathprobs)

        norm_pathprobs = []
        for i in pathprobs:
            norm_pathprobs.append(i/paths_sum)
       # print(norm_pathprobs, sum(norm_pathprobs))

        top18paths = []
        top18probs = []
        for i in range(0,len(norm_pathprobs)):
            if norm_pathprobs[i] > 0:
                top18probs.append(norm_pathprobs[i])
                top18paths.append(path_ids[i])

        print(len(list(set(top18paths))))
        top18paths, top18probs = zip(*sorted(zip(top18paths, top18probs)))
        fig,ax = plt.subplots()

        ax.bar(range(1,len(top18paths)+1), top18probs)
        print(top18paths)

        plt.savefig("forward_enumeration.pdf", format='pdf', dpi=300)

        sorted_pathlist = sorted(top18paths, key=lambda tup: tup[:])
        #print(sorted_pathlist)

gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
exhaustive_sampling = ExhaustiveSampling(gpm, outputfile=sys.argv[2],
                                     population_size=10000,
                                     reversibility=False, null_step=False)
exhaustive_sampling.all_paths()
# systematic_sampling.sample(sys.argv[3], 1234)