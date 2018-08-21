#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# Functions required for genotype-phenotype maps and sampling paths

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import sys
from math import e
import matplotlib.pyplot as plt
from scipy import sparse

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from gpmap import GenotypePhenotypeMap
from gpmap.utils import hamming_distance
from gpgraph.base import get_neighbors


def transition_matrix(gpm, population_size, minval=0, mutation_rate=1, null_steps=True, reversibility=True):
    """ADJACENCY MATRIX"""

    binary = list(gpm.binary)
    phenotypes = list(gpm.phenotypes)

    row, col, data, phen = [], [], [], []
    for bi in binary:
        neighbors = get_neighbors(bi, gpm.mutations)
        col_tmp = [binary.index(neighbor) for neighbor in neighbors]
        col += col_tmp
        row += [binary.index(bi)] * len(col_tmp)
        data += [1] * len(col_tmp)
        phen += [phenotypes[binary.index(bi)]] * len(col_tmp)

    ax = len(binary)
    P = sparse.csr_matrix((phen, (row, col)), shape=(ax, ax))

    """TRANSITION MATRIX"""
    """Phenotype ratio matrix"""

    P_T = P.T
    # dense
    R = P / P_T # dividing two sparse matrices returns dense matrix, why?
    #sparse
    R[np.isnan(R)] = 0
    R = sparse.csr_matrix(R)

    # bra = np.ones((len(gpm.binary), 1), int)
    # M = np.outer(gpm.phenotypes, bra)
    # print(M)
    # M = M * A

    # M_T = M.T
    # R = M / M_T
    # print(R)
    #
    # """Set nan to 0"""
    # R[np.isnan(R)] = 0
    #
    # R = sparse.csr_matrix(R)
    """Sparse matrix with 1 for all non-zero values of M"""
    I = R[:]
    I[I > 0] = 1
    I = sparse.csr_matrix(I)

    """FIXATION PROBABILITY MORAN"""

    """Exponential"""
    R_exp = R.power(population_size)

    """numerator"""
    R_mi = I - R

    """denominator"""
    R_exp_mi = I - R_exp

    """Put it all together"""
    T = R_mi / R_exp_mi  # outputs dense matrix, should be sparse
    T[np.isnan(T)] = 0

    """Normalize rows to 1"""
    Tn = sparse.spdiags(1. / T.sum(1).T, 0, *T.shape)
    T = Tn * T


    """Replace nan with 0"""
    T[np.isnan(T)] = 0

    return T, R


def transition_matrix_old(gpm_data, wildtype, mutations, population_size, minval=0, mutation_rate=1, null_steps=False, reversibility=False):
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
            neighbors = get_neighb(gpm_data, wildtype, current_state, mutations, reversibility)

            # Calculate fixation probability if the next state is a neighbor of the current state.
            if next_state in neighbors and next_state != current_state:
                # df.ix[row, column] = population_size * mutation_rate * max(0.,
                #                                                            fixation_probability(gpm_data, current_phen,
                #                                                                                 next_phen,
                #                                                                                 population_size))
                if population_size == 1:
                    df.ix[row, column] = 1/len(neighbors)
                else:
                    df.ix[row, column] = max(0., minval + fixation_probability_moran(gpm_data, current_phen, next_phen, population_size))/len(neighbors)

                # if current_phen < next_phen:
                #     pass

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


def get_neighb(gpm_data, wildtype, binarygenotype, mutations, reversibility=False):
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

def fixation_probability_moran(current, proposed, pop_size):
    # Calculate fixation probability.
    # fix_prob = 1 - e ** (-1 - rel_current / rel_proposed) / 1 - e ** (-pop_size * (1 - rel_current / rel_proposed)
    fix_prob = (1 - (current / proposed)) / (1 - (current / proposed)**pop_size)
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

def plot_pmf(pmf):
    # Plots probability distribution of a pmf, pmf is a dictionary(keys=paths, values=probabilities)
    paths = []
    probs = []
    for path, probability in pmf.items():
        paths.append(path)
        probs.append(probability)
    zipped = zip(paths, probs)
    sorted_paths = sorted(zipped, key=lambda x: x[0])
    num_of_paths = len(sorted_paths)
    f, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i for i in range(1, num_of_paths+1)],[pair[1] for pair in sorted_paths])
    ax.set_title("Probability Distribution of Paths")
    ax.set_ylabel("Probability")

    # Bar labels
    for label in range(1, num_of_paths+1):
        ax.text(label, -0.01, label, ha='center', va='top', fontsize=(300*(1/num_of_paths)))
    plt.xticks([])
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
    # tm = transition_matrix(gpm.data,
    #                        gpm.wildtype,
    #                        gpm.mutations,
    #                        population_size=10000,
    #                        mutation_rate=0.1,
    #                        null_steps=True,
    #                        reversibility=True)
    # print(tm)
    cmap = plt.cm.get_cmap('plasma')
    markers = ['--', '-bo', '-v', '-s', '-x', '-x']
    for i, pop_size in enumerate([1, 2, 5, 10, 50, 100, 868, 1000, 10000, 1000000]):
        x, y = [], []
        for proposed_fitn in np.arange(0.1, 5, 0.05):
            y.append(fixation_probability_moran(gpm.data, 1, proposed_fitn, pop_size=pop_size))
            x.append(proposed_fitn-1)
        y_at_xzero = y[x.index(min([abs(xi) for xi in x]))]
        plt.plot(x, y, markersize=2, color=cmap((1 / len(markers)) * i))
        plt.text(0, y_at_xzero, "%s" % pop_size)
    plt.axvline(0, linewidth=0.5, color="black", linestyle="--", zorder=0)
    plt.axhline(0, linewidth=0.5, color="black", linestyle="-", zorder=0)
    plt.show()
    plt.savefig("fixation_probability_plot.pdf", format='pdf', dpi=300)



