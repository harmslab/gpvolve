#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# master

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


class Sampling(object):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm=None, mutant=None):
        self.read_gpm(gpm)

        # Set mutant. Default: The furthest genotypes from wildtype by hamming-distance.
        if mutant == None:
            self.mutant = self.furthest_genotypes(self.wildtype, self.data.genotypes)[0]
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

        plt.plot(lag_intervals, convergence, '--bo')
        plt.savefig("%s_convergence.pdf" % self.outfilename, formar='pdf', dpi=300)


        # Take filename without extension.

        # Output masterlog as a .json file.
        with open("%s.json" % self.outfilename, 'w') as outfile:
            json.dump(masterlog, outfile, sort_keys=True, indent=4)

        return masterlog

    def propose_step(self, state_curr, reversibility):
        """ Propose the next step based on the current step's neighbors and whether reverse steps are allowed. """

        # Propose random step from neighbors of current genotype.
        state_prop = random.choice(self.get_neighbors(state_curr))
        # If reversibility is True, return the proposed state.
        if reversibility == True:
            pass
        # If reversibility is False, get signed hamming distance of the proposed genotype from the current genotype.
        elif reversibility == False:
            signed_hamming = self.signed_hamming_distance(state_curr, state_prop)
            # Keep proposing a new genotype while the signed hamming distance is negative (i.e. reverse step/step back)
            while signed_hamming <= 0:
                state_prop = random.choice(self.get_neighbors(state_curr))
                signed_hamming = self.signed_hamming_distance(state_curr, state_prop)
                continue

        return state_prop

    def signed_hamming_distance(self, current, proposed):
        """Return the signed Hamming distance between equal-length sequences """
        nonbinary = []

        # Get non-binary version of the genotypes
        for genotype in [current, proposed]:
            # First: get index of binary genotype in the dataframe.
            genotype_index = self.data.index[self.data['binary'] == genotype].tolist()
            # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
            nonbinary.append(self.data.iloc[genotype_index[0]]['genotypes'])

        # Count differences between wt and each genotype
        current_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(nonbinary[0], self.wildtype))
        proposed_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(nonbinary[1], self.wildtype))
        # Get the signed hamming distance between the two genotypes.
        # e.g. +1 if proposed states has one mutation more than the current state.
        signed_hamming = proposed_to_wt - current_to_wt

        return signed_hamming

    def furthest_genotypes(self, reference, genotypes):
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
                if self.signed_hamming_distance(binarygenotype, genotype) > 0:
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
            if self.null_steps == True:
                df.ix[row, row] = 1 - sum_of_prob

            # If residing in current state is not allowed, P(i->i) remains 0.
            # All other probabilities (P(i->j)) are adjusted so they sum to 1.
            elif self.null_steps == False:
                for column in range(0, len(df.ix[row, :])):
                    trans_probab = df.ix[row, column]

                    # Adjust probability so the sum of all P(i->j) equals 1.
                    try:
                        df.ix[row, column] = trans_probab/sum_of_prob
                    except ZeroDivisionError:
                        df.ix[row, column] = 0
        return df

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