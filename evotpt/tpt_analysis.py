#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach / Major parts copied from:
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/api.py and
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/analysis/sparse

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import sys
import numpy as np
from scipy import sparse



# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt.visualization import GenotypePhenotypeGraph
from evotpt import utils
from gpmap import GenotypePhenotypeMap

def number_of_paths(pmf):
    nb_paths = len(pmf)
    return nb_paths

def adaptive_paths(gpm, pmf):
    ap_pmf = {}
    for path, prob in pmf.items():
        phens = [utils.get_phenotype(gpm.data, gt) for gt in path]
        # if the minimum phenotype is not at position 0, then the path is not adaptive
        minm = min(phens)
        minima = [i for i, x in enumerate(phens) if x == minm]
        if phens.index(min(phens)) == 0 and len(minima) == 1:
            ap_pmf[path] = prob
    return ap_pmf

def non_adaptive_paths(pmf, ap_pmf):
    non_ap_pmf = {}
    for path, prob in pmf.items():
        if path not in ap_pmf:
            non_ap_pmf[path] = prob
    return non_ap_pmf

def sinks(ratio_matrix):
    R = ratio_matrix[:]
    R[R < 1] = 0
    sinks = []
    for i, row in enumerate(R):
        if R[i].sum(axis=1) == 0:
            sinks.append(i)
    return sinks

def peaks(ratio_matrix):
    R = ratio_matrix[:]
    R[R > 1] = 0
    peaks = []
    for i, row in enumerate(R):
        if R[i].sum(axis=1) == 0:
            peaks.append(i)
    return peaks

def chains(ratio_matrix):
    R = ratio_matrix[:]
    R[R > 1] = 0
    chains = []
    print(R)
    for i, row in enumerate(R):
        print(R[i])
        if R[i].count_nonzero() == 1:
            # get edge: row = 1, column = the nonzero entry of row R[i]
            chains.append((i, R[i].nonzero()[1][0]))
    return chains

def path_difference(pmf, fraction=1):
    f_paths = []
    flux = 0
    for path, prob in pmf.items():
        if flux < fraction:
            flux += prob
            f_paths.append(path)

    # Get length of most probable path
    path_lngh = len(f_paths[0])
    # Get length of each genotype
    gt_lngh = len(list(f_paths[0][0]))
    # Only consider paths of that length
    paths = [path for path in f_paths if len(path) == path_lngh]
    path_smlty = {}
    for step in range(0, path_lngh):
        binaries = [path[step] for path in paths]
        """Create # of sites x # genotypes matrix"""
        N = np.array([list(binary) for binary in binaries])
        N = N.astype(int)

        """Create empty adjacency matrix"""
        X = np.empty([len(f_paths),len(f_paths)], int)

        """Substract N from each row of N to get the hamming distance of genotype x (N[x]) to all other genotypes"""
        for i in range(0, len(binaries)):
            row = np.absolute(N[i] - N)
            X[i] = row.sum(axis=1)

        avg_div_per_gt = X.sum(axis=1)/len(f_paths)

        path_smlty[step] = (sum(avg_div_per_gt)/len(avg_div_per_gt))/gt_lngh

    return path_smlty

def length_distr(pmf):
    paths = [*pmf]
    lengths = [len(path) for path in paths]
    length_distr = dict((x, lengths.count(x)) for x in lengths)
    return length_distr

def non_adaptive_flux(pmf, ap_pmf):
    flux = sum(pmf.values()) - sum(ap_pmf.values())
    return flux


def path_divergence_hammingdist(gpm, pmf):
    pass


