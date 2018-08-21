#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach / Major parts copied from:
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/api.py and
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/analysis/sparse

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import sys
import numpy as np
import networkx as nx



# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt import utils
from gpmap import GenotypePhenotypeMap

def to_edges(paths):
    """Get list of unique edges in a list or dictionary of paths"""
    # If paths is dictionary, get list of keys.
    if isinstance(paths, dict):
        pathslist = list(paths.keys())

    edges = []
    for path in pathslist:
        for i in range(0, len(path)-1):
            # Store edges as a list of tuples.
            edges.append((path[i], path[i+1]))

    # Get unique list of edges.
    uniq = set(edges)
    return uniq


def number_of_paths(pmf):
    nb_paths = len(pmf)
    return nb_paths


def adaptive_paths(G, pmf, get_edges=False):
    ap_pmf = {}
    edges = []
    for path, prob in pmf.items():
        phens = [nx.get_node_attributes(G, "phenotypes")[node] for node in path]
        # if the minimum phenotype is not at position 0, then the path is not adaptive
        adaptive = True
        for i, ph in enumerate(phens):
            if ph != phens[-1] and phens[i+1] < ph:
                adaptive = False
                break

        if adaptive == True:
            ap_pmf[path] = prob
    return ap_pmf

def non_adaptive_paths(pmf, ap_pmf):
    non_ap_pmf = {}
    if ap_pmf == False:
        return pmf
    else:
        for path, prob in pmf.items():
            if path not in ap_pmf:
                non_ap_pmf[path] = prob
        return non_ap_pmf


def sinks(ratio_matrix):
    """Take a ratio matrix (Tij = i/j) and find sinks"""
    R = ratio_matrix.copy()
    # Only keep ratios i/j above 1, i.e. downhill moves.
    R[R < 1] = 0
    sinks = []
    for i, row in enumerate(R):
        # If there are no possible downhill moves we are at a sink.
        if R[i].sum(axis=1) == 0:
            sinks.append(i)
    return sinks


def peaks(ratio_matrix):
    """Take a ratio matrix (Tij = i/j) and find sinks"""
    R = ratio_matrix.copy()
    # Only keep ratios i/j below 1, i.e. uphill moves.
    R[R > 1] = 0
    peaks = []
    for i, row in enumerate(R):
        # If there are no uphill moves, we are at a peak.
        if R[i].sum(axis=1) == 0:
            peaks.append(i)
    return peaks


def chains(ratio_matrix):
    """Take a ratio matrix (Tij = i/j) and find chains"""
    R = ratio_matrix[:]
    R[R >= 1] = 0
    chains = []
    for i, row in enumerate(R):
        if R[i].count_nonzero() == 1:
            # get edge: row = 1, column = the nonzero entry of row R[i]
            chains.append((i, R[i].nonzero()[1][0]))
    return chains


def spatial_path_divergence(G, paths):
    pass


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

        div_per_gt = X.sum(axis=1)
        avg_div_per_gt = div_per_gt/len(f_paths)

        path_smlty[step+1] = (sum(avg_div_per_gt)/len(avg_div_per_gt))/gt_lngh

    return path_smlty


def length_distr(pmf):
    paths = [*pmf]
    lengths = [len(path) for path in paths]
    length_distr = dict((x, lengths.count(x)) for x in lengths)
    return length_distr


def non_adaptive_flux(pmf, ap_pmf):
    flux = sum(pmf.values()) - sum(ap_pmf.values())
    return flux
