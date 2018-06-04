#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach
#
# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import pandas as pd
import json
import sys
import random
from math import e
from itertools import permutations


# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from gpmap.utils import encode_mutations, construct_genotypes, DNA, hamming_distance, genotypes_to_binary
from gpgraph.base import get_neighbors

try:
    # Check if argument 1 is given.
    infile = sys.argv[1]
    if infile.endswith(".json"):
        # Check if input file is a .json file.
        gpmapfile = infile
        # Get filename without file extension and path.
        # filename = infile.split("/")[-1].split(".")[-2]
    else:
        print("Wrong file format: Provide .json file.")
        sys.exit(1)
except IndexError:
    print("Missing first argument: Provide .json file.")
    sys.exit(1)

outputfile = sys.argv[2]

# Load .json file.
with open(gpmapfile, "r") as json_file:
    gpm = json.load(json_file)

# Assign genotype-phenotype-map objects.
wt, genotypes, phenotypes, mutations, stdev = gpm["wildtype"], gpm["genotypes"],\
                                                          gpm["phenotypes"], gpm["mutations"],\
                                                          gpm["stdeviations"]

# Use gpmap.utils.genotypes_to_binary to get binary phenotypes.
binaries = genotypes_to_binary(wt, genotypes, mutations)
binary_wildtype = binaries[genotypes.index(wt)]

"""MONTE CARLO SIMULATION"""

class MonteCarlo(object):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""

    def __init__(self, genotypes, phenotypes,
                 mutations, binaries, wildtype,
                 binary_wildtype, filename, population_size, random_seed):

        # Store data in dataframe.
        data = dict(genotypes=genotypes,
                    phenotypes=phenotypes,
                    binaries=binaries)
        self.data = pd.DataFrame(data)

        # Set mutations.
        self._mutations = mutations

        # Set wildtype.
        self._wildtype = wildtype

        # Set binary wildtype.
        self.binary_wt = binary_wildtype

        # Set filename.
        self._filename = filename

        # Set population size.
        self.pop_size = population_size

        # Set random seed.
        self.rand_seed = random_seed

        # Create dictionary with neighbours for each genotype.
        self.NeighborsDict = self.get_neighbors(self.data.binaries)

    def sample(self, iterations, random_seed):
        """Sample paths by iterating self.simulate. Save output as .json file"""
        masterlog = {}
        random.seed(random_seed)
        masterlog["random_seed"] = random_seed
        # Iterate self.sample and add every resulting path to masterlog.
        for i in range(1, iterations+1):
            log = self.simulate()
            masterlog[str(i)] = self.simulate()

            # Get the index of every genotype from self.data['genotypes'] and concatenate to get a unique path ID.
            # (corresponds to genotype position in data set .json file).
            path_id =[]
            for genotype in log["path"]:
                # Get index of genotype.
                genotype_id = self.data.index[self.data['genotypes'] == genotype].tolist()
                path_id.append(genotype_id[0])
            # make path ID a tuple and add to masterlog.
            path_id = tuple(path_id)
            masterlog[str(i)]["path_id"] = [path_id]

            print("Path %s:" % i, masterlog[str(i)]["path"])

        # Take filename without extension.
        filename = self._filename.split(".")[0]
        # Output masterlog as a .json file.
        with open(sys.argv[2], 'w') as outfile:
            json.dump(masterlog, outfile, sort_keys=True, indent=4)

        return masterlog

        return masterlog

    def simulate(self):
        """Monte Carlo simulation"""

        stepcount = 0
        StateCurr = self.binary_wt

        log = {}
        log["attempts"] = [0]
        log["path"] = []

        while stepcount < len(binary_wildtype):
            log["attempts"][0] += 1
            # Proposes random new state from neighbors of the current state.
            StateProp = random.choice(self.NeighborsDict[StateCurr])

            # Get phenotypes of current and proposed genotype.
            PhenoCurr, PhenoProp = self.get_phenotype(StateCurr), self.get_phenotype(StateProp)

            # Get random number from a uniform distribution between 0 and 1.
            rand_unif = random.uniform(0, 1)

            if self.fixation_probability(PhenoCurr, PhenoProp, self.pop_size) > rand_unif:
                # Log current genotype
                log["path"].append(StateCurr)
                # Proposed state is accepted.
                StateCurr = StateProp
                stepcount += 1

            else:
                # Propose a different step instead, i.e. start over.
                continue
        # Log the last genotype.
        log["path"].append(StateCurr)
        return log

    def hamming_distance(self, s1, s2):
        """Return the signed Hamming distance between equal-length sequences"""
        s1ints, s2ints = [], []
        for c1, c2 in zip(s1, s2):
            s1ints.append(int(c1))
            s2ints.append(int(c2))
        hammingdirection = sum(s1ints) - sum(s2ints)

        abs = sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

        return abs * hammingdirection

    # def hamming_distance(self, s1, s2):
    #     """Return the Hamming distance between equal-length sequences"""
    #     sum1 = sum(int(ch1) for ch1 in s1)
    #     sum2 = sum(int(ch2) for ch2 in s2)
    #     return sum1-sum2

    def get_neighbors(self, binaries):
        """Get only the neighbors with a positive hamming distance, i.e. steps forward/adding a mutation"""
        neighbors = {}
        for ref_binary in binaries:
            neighbors[ref_binary] = []
            for binary in binaries:
                if self.hamming_distance(binary, ref_binary) == 1:
                    neighbors[ref_binary].append(binary)
        return neighbors

    def get_phenotype(self, binary):
        """Return phenotype for a certain genotype"""
        # Get row index of phenotype in self.data.
        gt_index = self.data.index[self.data['binaries'] == binary].tolist()
        # Pull out the corresponding phenotype from self.data using the genotype index.
        phenotype = self.data.iloc[gt_index[0]]['phenotypes']
        return phenotype

    def max_phenotype(self):
        """Get the highest phenotype"""
        max_phen = max(self.data.phenotypes)
        return max_phen

    def relative_phenotype(self, phenotype):
        """Calculate the relative phenotype using the highest phenotypes as reference"""
        rel_phen = phenotype/self.max_phenotype()
        return rel_phen

    def fixation_probability(self, phe1, phe2, pop_size):
        """Calculate the fixation probability based on a model by Gillespie, Gillespie, 2010, JHU press."""
        # Get relative phenotypes.
        rel_phe1, rel_phe2 = self.relative_phenotype(phe1), self.relative_phenotype(phe2)
        # Calculate fixation probability.
        fix_prob = 1 - e ** (- 1-rel_phe1/rel_phe2) / 1 - e ** (- pop_size * (1-rel_phe1/rel_phe2))
        round_fix_prob = round(fix_prob, 10)
        return fix_prob


sampling = MonteCarlo(genotypes, phenotypes, mutations,
                      binaries, wt, binary_wildtype,
                      outputfile, 10**5, 1234)

print(sampling.sample(100, 1234))


### get_neighbors not right yet. Defined as hamming distance of one but that's not right for sites with multiple
### possible mutations

### no parallel or backward steps, stop after

### use get_neighbors function from gpgrap.base; Only does it for non-binary genotypes(?)


# MISCELLANEOUS:
# gpmap = GenotypePhenotypeMap(wt, # wildtype sequence
#                    genotypes, # genotypes
#                    phenotypes, # phenotypes
#                    stdeviations=None, # errors in measured phenotypes
#                    log_transform=False, # Should the map log_transform the space?
#                    mutations=mutations # Substitution map to alphabet
#                            )
# neighbors = {}
# for genotype in gpmap.genotypes:
#     neighbors[genotype] = get_neighbors(genotype, gpmap.mutations)
# print(neighbors)