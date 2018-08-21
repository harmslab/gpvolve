#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# -------------------------------------------------------------------------
# USAGE: montecarlo_path_sampling.py <dataset>.json <outputfile>.json <iteration number>
# -------------------------------------------------------------------------

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


class MarkovChainMonteCarlo(object):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm, filename, population_size, reversibility=True, allow_resting=False, **kwargs):

        self.add_gpm(gpm)

        # Set filename.
        self._filename = filename

        # Set population size.
        self.pop_size = population_size

        # Set reversibility
        self.reversibility = reversibility

        # Set allow restinf
        self.allow_resting = allow_resting

        print("Reverse steps allowed: %s. Default: True" % reversibility)

    def add_gpm(self, gpm):
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

        self.farthest_gts = self.farthest_genotypes(self.wildtype, self.data.genotypes)



    def transition_matrix(self):
        """ Create transition NxN matrix where N is the number of genotypes """

        data = self.data
        # Create transition matrix, column names and row indices are genotype names.
        df = pd.DataFrame(index=list(data.genotypes), columns=list(data.genotypes))

        # Loop over rows.
        for row in range(0, len(df.ix[:, 0])):

            # Loop over columns.
            for column in range(0, len(df.ix[row, :])):

                # Get phenotypes of phenotypes
                current_state = df.index[row]
                next_state = list(df)[column]
                current_phen = self.get_phenotype(current_state)
                next_phen = self.get_phenotype(next_state)

                # Calculate fixation probability if the next state is a neighbor of the current state.
                if next_state in self.get_neighbors(current_state):
                    df.ix[row, column] = max(0, self.fixation_probability(current_phen, next_phen, self.pop_size))
                    if current_phen < next_phen:
                        pass

                # If next state is not a neighbor, transition probability is 0.
                else:
                    df.ix[row, column] = 0

        # Update probabilities depending on whether residing at current state is allowed.
        # Loop over rows
        for row in range(0, len(df.ix[:, 0])):

            # Get sum of probabilities in row. P(i->i) is still set to 0.
            sum_of_prob = sum(column for column in df.ix[row, :])

            # If residing in current state is allowed, P(i->i), i.e. remaining in current state, is 1 - sum_of_prob.
            if self.allow_resting == True:
                df.ix[row, row] = 1 - sum_of_prob

            # If residing in current state is not allowed, P(i->i) remains 0.
            # All other probabilities (P(i->j)) are adjusted so they sum to 1.
            elif self.allow_resting == False:
                for column in range(0, len(df.ix[row, :])):
                    trans_probab = df.ix[row, column]

                    # Adjust probability so the sum of all P(i->j) equals 1.
                    try:
                        df.ix[row, column] = trans_probab/sum_of_prob
                    except ZeroDivisionError:
                        df.ix[row, column] = 0

        return df

    def sample_from_tm(self):
        tm = self.tm

        # print(transition_matrix.ix[0, :])
        genotypes = list(tm)
        gt_indices = [index for index in range(0, len(genotypes))]
        StateCurr = self.binary_wildtype

        log = {}
        log["attempts"] = [0]
        log["path"] = []

        while StateCurr not in self.farthest_gts:

            pmf = [probability for probability in tm.ix[genotypes.index(StateCurr), :]]
            sample = genotypes[rv_discrete(name='sample', values=(gt_indices, pmf)).rvs()]
            StateCurr = sample

            log["path"].append(StateCurr)

        return log


    def sample(self, iterations, random_seed):
        """Sample paths by iterating self.simulate. Save output as .json file"""
        masterlog = {}
        random.seed(random_seed)
        masterlog["random_seed"] = random_seed

        self.tm = self.transition_matrix()

        # Iterate self.sample and add every resulting path to masterlog.
        for i in range(1, iterations+1):
            log = self.sample_from_tm()
            masterlog[str(i)] = self.sample_from_tm()

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
        state_prop = random.choice(self.get_neighbors(state_curr))
        # If reversibility is True, return the proposed state.
        if reversibility == True:
            pass
        # If reversibility is False, get signed hamming distance of the proposed genotype from the current genotype.
        elif reversibility == False:
            signed_hamming = self.signed_hamming_distance(self.wildtype, state_curr, state_prop)
            # Keep proposing a new genotype while the signed hamming distance is negative (i.e. reverse step/step back)
            while signed_hamming <= 0:
                state_prop = random.choice(self.get_neighbors(state_curr))
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

    def get_neighbors(self, binarygenotype):
        """ (Adapted from gpgraph.base)

        - Takes a binary genotype from self.simulate and the mutation dictionary as arguments.

        - Gets non-binary version of genotype.

        - Gets neighbors of genotype using mutations dictionary.

        - Turns non-binary neighbors into binary and returns as tuple.

        """
        # Set mutations.
        mutations = self.mutations

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

        if self.reversibility == False:
            irrev_binarymutants = []
            wildtype = self.wildtype
            for genotype in binaryneighbors:
                if self.signed_hamming_distance(wildtype, binarygenotype, genotype) > 0:
                    irrev_binarymutants.append(genotype)
            binaryneighbors = irrev_binarymutants
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



sampling = MarkovChainMonteCarlo(gpm, outputfile, 10000, reversibility=True, allow_resting=False)

sampling.sample(int(sys.argv[3]), 1234)


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