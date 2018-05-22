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

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from gpmap.utils import encode_mutations, construct_genotypes, DNA, hamming_distance, genotypes_to_binary

try:
    # Check if argument 1 is given.
    infile = sys.argv[1]
    if infile.endswith(".json"):
        # Check if input file is a .json file.
        gpmapfile = infile
        # Get filename without file extension and path.
        filename = infile.split("/")[-1].split(".")[-2]
    else:
        print("Wrong file format: Provide .json file.")
        sys.exit(1)
except IndexError:
    print("Missing first argument: Provide .json file.")
    sys.exit(1)

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
                 binaries, wildtype, binary_wildtype,
                 filename, population_size, random_seed):

        # Store data in dataframe.
        data = dict(genotypes=genotypes,
                    phenotypes=phenotypes,
                    binaries=binaries)
        self.data = pd.DataFrame(data)

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

    def sample(self, iterations):
        """Sample paths by iterating self.simulate. Save output as .json file"""
        masterlog = {}
        masterlog["random_seed"] = self.rand_seed
        # Iterate self.sample.
        for i in range(1, iterations):
            masterlog[str(i)] = self.simulate()
            print("Path %s:" % i, masterlog[str(i)]["path"])

        # Output masterlog as a .json file.
        with open('%s_monte-carlo_log.json' % filename, 'w') as outfile:
            json.dump(masterlog, outfile, sort_keys=True, indent=4)

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
                # Proposed state is accepted.
                StateCurr = StateProp
                log["path"].append(StateCurr)
                stepcount += 1

            else:
                # Propose a different step instead, i.e. start over.
                continue
        return log

    def hamming_distance(self, s1, s2):
        """Return the Hamming distance between equal-length sequences"""
        sum1 = sum(int(ch1) for ch1 in s1)
        sum2 = sum(int(ch2) for ch2 in s2)
        return sum1-sum2

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
        return fix_prob


sampling = MonteCarlo(genotypes, phenotypes, binaries, wt, binary_wildtype, filename, 10**5, 1234)

print(sampling.sample(10))

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