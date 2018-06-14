#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# Functions required for genotype-phenotype maps and sampling paths

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


def transition_matrix(gpm_data, wildtype, mutations, population_size, mutation_rate=1, null_steps=False, reversibility=False):
    """ Create transition NxN matrix where N is the number of genotypes """

    data = gpm_data
    # Create transition matrix, column names and row indices are genotype names.
    df = pd.DataFrame(index=list(data.genotypes), columns=list(data.genotypes))

    # Loop over rows.
    for row in range(0, len(df.ix[:, 0])):

        # Loop over columns.
        for column in range(0, len(df.ix[row, :])):

            # Get phenotypes of phenotypes
            current_state = df.index[row]
            next_state = list(df)[column]
            current_phen = get_phenotype(gpm_data, current_state)
            next_phen = get_phenotype(gpm_data, next_state)
            neighbors = get_neighbors(gpm_data, wildtype, current_state, mutations, reversibility)

            # Calculate fixation probability if the next state is a neighbor of the current state.
            if next_state in neighbors:
                #df.ix[row, column] = population_size * mutation_rate * max(0.,
                                                                           # fixation_probability(gpm_data, current_phen,
                                                                           #                      next_phen,
                                                                           #                      population_size))
                df.ix[row, column] = max(0., fixation_probability_moran(gpm_data, current_phen, next_phen, population_size))/len(neighbors)
                if current_phen < next_phen:
                    pass

            # If next state is not a neighbor, transition probability is 0.
            else:
                df.ix[row, column] = 0.

    # Update probabilities depending on whether residing at current state is allowed.
    # Loop over rows
    for row in range(0, len(df.ix[:, 0])):

        # Get sum of probabilities in row. P(i->i) is still set to 0.
        sum_of_prob = sum(column for column in df.ix[row, :])

        # If residing in current state is allowed, P(i->i), i.e. remaining in current state, is 1 - sum_of_prob.
        if null_steps == True:
            df.ix[row, row] = max(0., 1 - sum_of_prob)

        # If residing in current state is not allowed, P(i->i) remains 0.
        elif null_steps == False:
            pass

        sum_of_prob = sum(column for column in df.ix[row, :])

        # All probabilities are adjusted so they sum to 1.
        for column in range(0, len(df.ix[row, :])):
            trans_probab = df.ix[row, column]

            # Adjust probability so the sum of all P(i->j) equals 1.
            try:
                df.ix[row, column] = trans_probab/sum_of_prob
            except ZeroDivisionError:
                df.ix[row, column] = 0.
    return df


def get_neighbors(gpm_data, wildtype, binarygenotype, mutations, reversibility=False):
    """ (Adapted from gpgraph.base)

    - Takes a binary genotype from self.simulate and the mutation dictionary as arguments.

    - Gets non-binary version of genotype.

    - Gets neighbors of genotype using mutations dictionary.

    - Turns non-binary neighbors into binary and returns as tuple.

    """
    # Get non-binary version of genotype:
    # First: get index of genotype in the dataframe.
    binarygenotype_index = gpm_data.index[gpm_data['binary'] == binarygenotype].tolist()
    # Second: Pull out the corresponding binary from the dataframe and set as binary wildtype.
    genotype = gpm_data.iloc[binarygenotype_index[0]]['genotypes']

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
        try:
            # Get binary version of neighbor:
            # First: get index of neighbor in the dataframe.
            neighbor_index = gpm_data.index[gpm_data['genotypes'] == neighbor].tolist()
            # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
            binaryneighbors.append(gpm_data.iloc[neighbor_index[0]]['binary'])
        except IndexError:
            pass

    if reversibility == False:
        irrev_binarymutants = []
        for genotype in binaryneighbors:
            if signed_hamming_distance(wildtype, binarygenotype, genotype) > 0:
                irrev_binarymutants.append(genotype)
        binaryneighbors = irrev_binarymutants
    # Return neighbors as tuple.
    return tuple(binaryneighbors)

def fixation_probability(gpm_data, current, proposed, pop_size):
    """Calculate the fixation probability based on a model by Gillespie, Gillespie, 2010, JHU press."""
    # Get relative phenotypes.
    rel_current, rel_proposed = relative_phenotype(gpm_data, current), relative_phenotype(gpm_data, proposed)
    # Calculate fixation probability.
    # fix_prob = 1 - e ** (-1 - rel_current / rel_proposed) / 1 - e ** (-pop_size * (1 - rel_current / rel_proposed)
    fix_prob = 1 - e ** -((rel_proposed / rel_current) - 1) / 1 - e ** -pop_size * ((rel_proposed / rel_current) - 1)
    # print("Current: %s, Proposed: %s\nFixation Probability: %s" % (rel_current, rel_proposed, fix_prob))
    return fix_prob

def fixation_probability_moran(gpm_data, current, proposed, pop_size):
    """Calculate the fixation probability based on a model by Gillespie, Gillespie, 2010, JHU press."""
    # Get relative phenotypes.
    rel_current, rel_proposed = relative_phenotype(gpm_data, current), relative_phenotype(gpm_data, proposed)

    # Calculate fixation probability.
    # fix_prob = 1 - e ** (-1 - rel_current / rel_proposed) / 1 - e ** (-pop_size * (1 - rel_current / rel_proposed)
    fix_prob = (1 - (rel_current / rel_proposed)) / (1 - (rel_current / rel_proposed)**pop_size)
    # print("Current: %s, Proposed: %s\nFixation Probability: %s" % (rel_current, rel_proposed, fix_prob))
    return fix_prob

def get_phenotype(gpm_data, binary):
    """Return phenotype for a certain genotype"""
    # Get row index of phenotype in self.data.
    gt_index = gpm_data.index[gpm_data['binary'] == binary].tolist()
    # Pull out the corresponding phenotype from self.data using the genotype index.
    phenotype = gpm_data.iloc[gt_index[0]]['phenotypes']
    return phenotype


def furthest_genotypes(gpm_data, reference, genotypes):
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
        mutant_index = gpm_data.index[gpm_data['genotypes'] == mutant].tolist()
        # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
        binarymutants.append(gpm_data.iloc[mutant_index[0]]['binary'])

    return binarymutants


def max_phenotype(gpm_data):
    """Get the highest phenotype"""
    max_phen = max(gpm_data.phenotypes)
    return max_phen

def min_phenotype(gpm_data):
    """Get the highest phenotype"""
    min_phen = min(gpm_data.phenotypes)
    return min_phen

def relative_phenotype(gpm_data, phenotype):
    """Calculate the relative phenotype using the highest phenotypes as reference"""
    rel_phen = phenotype/max_phenotype(gpm_data)
    return rel_phen


def signed_hamming_distance(wildtype, current, proposed):
    """Return the signed Hamming distance between equal-length sequences """
    nonbinary = []

    # # Get non-binary version of the genotypes
    # for genotype in [current, proposed]:
    #     # First: get index of binary genotype in the dataframe.
    #     genotype_index = self.data.index[self.data['binary'] == genotype].tolist()
    #     # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
    #     nonbinary.append(self.data.iloc[genotype_index[0]]['genotypes'])

    # Count differences between wt and each genotype
    current_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(current, wildtype))
    proposed_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(proposed, wildtype))
    # Get the signed hamming distance between the two genotypes.
    # e.g. +1 if proposed states has one mutation more than the current state.
    signed_hamming = proposed_to_wt - current_to_wt

    return signed_hamming

def unique_sorted_paths(paths):
    """ Return unique sorted list of path IDs """
    uniq = list(set(paths))
    uniq_sorted_pathlist = sorted(uniq, key=lambda tup: tup[:])
    return uniq_sorted_pathlist