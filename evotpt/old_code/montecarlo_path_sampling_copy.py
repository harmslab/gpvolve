#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# -------------------------------------------------------------------------
# USAGE: montecarlo_path_sampling.py <dataset>.json <outputfile>.json <iteration number>
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import json
import sys
import random
from math import e

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from gpmap import GenotypePhenotypeMap
from gpmap.utils import hamming_distance


try:
    # Check if argument 1 is given.
    infile = sys.argv[1]
    if infile.endswith(".json"):
        # Check if input file is a .json file.
        gpmapfile = infile
        # Get filename without file extension and path.
        print(infile.split("/")[-1].split(".")[-2])
    else:
        print("Wrong file format: Provide .json file.")
        sys.exit(1)
except IndexError:
    print("Missing first argument: Provide .json file.")
    sys.exit(1)

outputfile = sys.argv[2]

# Load .json file.
with open(gpmapfile, "r") as json_file:
    gpmapfile = json.load(json_file)

# Assign genotype-phenotype-map objects.
wt, genotypes, phenotypes, mutations, stdev = gpmapfile["wildtype"], gpmapfile["genotypes"],\
                                                          gpmapfile["phenotypes"], gpmapfile["mutations"],\
                                                          gpmapfile["stdeviations"]

gpm = GenotypePhenotypeMap(wt, # wildtype sequence
                   genotypes, # genotypes
                   phenotypes, # phenotypes
                   stdeviations=None, # errors in measured phenotypes
                   log_transform=False, # Should the map log_transform the space?
                   mutations=mutations # Substitution map to alphabet
                           )

"""MONTE CARLO SIMULATION"""


class MonteCarlo(object):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm, filename, population_size, reversibility=True, **kwargs):

        self.add_gpm(gpm)

        # Set filename.
        self._filename = filename

        # Set population size.
        self.pop_size = population_size

        # Set reversibility
        self.reversibility = reversibility

        print("Reverse steps allowed: %s. Default: True" % reversibility)

    def add_gpm(self, gpm):
        """Assign GenotypePhenotypeMaps properties to variables"""
        # Add gpm
        self.gpm = gpm
        self.data = self.gpm.data

        # Set wildtype
        self.wildtype = gpm._wildtype

        # Get binary of wildtype:
        # First: get index of wildtype in the dataframe.
        genotype_index = self.data.index[self.data['genotypes'] == self.wildtype].tolist()
        # Second: Pull out the corresponding binary from the dataframe and set as binary wildtype.
        self.binary_wildtype = self.data.iloc[genotype_index[0]]['binary']

        self.farthest_gts = self.farthest_genotypes(self.wildtype, self.data.genotypes)

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

    def simulate(self):
        """Monte Carlo simulation"""

        stepcount = 0
        StateCurr = self.binary_wildtype

        log = {}
        log["attempts"] = [0]
        log["path"] = []

        while StateCurr not in self.farthest_gts:

            log["attempts"][0] += 1

            # Proposes random new state from neighbors of the current state.
            StateProp = self.propose_step(StateCurr, self.reversibility)

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

    def propose_step(self, state_curr, reversibility):
        # Propose random step from neighbors of current genotype.
        state_prop = random.choice(self.get_neighbors(state_curr, gpm.mutations))
        # If reversibility is True, return the proposed state.
        if reversibility == True:
            pass
        # If reversibility is False, get signed hamming distance of the proposed genotype from the current genotype.
        elif reversibility == False:
            signed_hamming = self.signed_hamming_distance(self.wildtype, state_curr, state_prop)
            # Keep proposing a new genotype while the signed hamming distance is negative (i.e. reverse step/step back)
            while signed_hamming <= 0:
                state_prop = random.choice(self.get_neighbors(state_curr, gpm.mutations))
                signed_hamming = self.signed_hamming_distance(self.wildtype, state_curr, state_prop)
                continue

        return state_prop

    def signed_hamming_distance(self, wt, current, proposed):
        """Return the signed Hamming distance between equal-length sequences """
        nonbinary = []

        # Get non-binary version of the genotypes
        for genotype in [current, proposed]:
            # First: get index of binary genotype in the dataframe.
            genotype_index = self.data.index[self.data['binary'] == genotype].tolist()
            # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
            nonbinary.append(self.data.iloc[genotype_index[0]]['genotypes'])

        # Count differences between wt and each genotype
        current_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(nonbinary[0], wt))
        proposed_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(nonbinary[1], wt))
        # Get the signed hamming distance between the two genotypes.
        # e.g. +1 if proposed states has one mutation more than the current state.
        signed_hamming = proposed_to_wt - current_to_wt

        return signed_hamming

    def farthest_genotypes(self, reference, genotypes):
        """Find the genotypes in the system that differ at the most sites. """
        mutations = 0
        mutants = []
        mutantsx = []
        for genotype in genotypes:
            differs = hamming_distance(genotype, reference)
            if differs > mutations:
                mutations = int(differs)
                mutantsx = mutants[:]
                mutantsx.append(str(genotype))
            elif differs == mutations:
                mutantsx.append(str(genotype))

        binarymutants = []
        for mutant in mutantsx:
            # Get binary version of neighbor:
            # First: get index of neighbor in the dataframe.
            mutant_index = self.data.index[self.data['genotypes'] == mutant].tolist()
            # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
            binarymutants.append(self.data.iloc[mutant_index[0]]['binary'])
        return binarymutants

    def get_neighbors(self, binarygenotype, mutations):
        """ (Adapted from gpgraph.base)

        - Takes a binary genotype from self.simulate and the mutation dictionary as arguments.

        - Gets non-binary version of genotype.

        - Gets neighbors of genotype using mutations dictionary.

        - Turns non-binary neighbors into binary and returns as tuple.

        """

        # Get non-binary version of genotype:
        # First: get index of genotype in the dataframe.
        binarygenotype_index = self.data.index[self.data['binary'] == binarygenotype].tolist()
        # Second: Pull out the corresponding binary from the dataframe and set as binary wildtype.
        genotype = self.data.iloc[binarygenotype_index[0]]['genotypes']

        neighbors = tuple()
        binaryneighbors = []

        for i, char in enumerate(genotype):
            # Copy reference genotype
            genotype2 = list(genotype)[:]

            # Find possible mutations at site i.
            options = mutations[i][:]
            options.remove(char)

            # Construct neighbor genotypes.
            for j in options:
                genotype2[i] = j
                genotype2_ = "".join(genotype2)
                neighbors += (genotype2_,)

        for neighbor in neighbors:
            # Get binary version of neighbor:
            # First: get index of neighbor in the dataframe.
            neighbor_index = self.data.index[self.data['genotypes'] == neighbor].tolist()
            # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
            binaryneighbors.append(self.data.iloc[neighbor_index[0]]['binary'])

        # Return neighbors as tuple.
        return tuple(binaryneighbors)

    def get_phenotype(self, binary):
        """Return phenotype for a certain genotype"""
        # Get row index of phenotype in self.data.
        gt_index = self.data.index[self.data['binary'] == binary].tolist()
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

    def fixation_probability(self, current, proposed, pop_size):
        """Calculate the fixation probability based on a model by Gillespie, Gillespie, 2010, JHU press."""
        # Get relative phenotypes.
        rel_current, rel_proposed = self.relative_phenotype(current), self.relative_phenotype(proposed)
        # Calculate fixation probability.
        # fix_prob = 1 - e ** (-1 - rel_current / rel_proposed) / 1 - e ** (-pop_size * (1 - rel_current / rel_proposed)
        fix_prob = 1 - e ** -((rel_proposed / rel_current) - 1) / 1 - e ** -pop_size * ((rel_proposed / rel_current) - 1)
        # print("Current: %s, Proposed: %s\nFixation Probability: %s" % (rel_current, rel_proposed, fix_prob))
        return fix_prob


sampling = MonteCarlo(gpm, outputfile, 10, reversibility=False)
sampling.sample(int(sys.argv[3]), 1234)


### get_neighbors not right yet. Defined as hamming distance of one but that's not right for sites with multiple
### possible mutations

### no parallel or backward steps, stop after

### use get_neighbors function from gpgrap.base; Only does it for non-binary genotypes(?)


# MISCELLANEOUS:
#
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


# def signed_hamming_distance(self, s1, s2):
    #     """Return the signed Hamming distance between equal-length sequences"""
    #     s1ints, s2ints = [], []
    #     for c1, c2 in zip(s1, s2):
    #         s1ints.append(int(c1))
    #         s2ints.append(int(c2))
    #     hammingdirection = sum(s1ints) - sum(s2ints)
    #
    #     abs = sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
    #
    #     return abs * hammingdirection