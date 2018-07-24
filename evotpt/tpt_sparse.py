#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach / Major parts copied from:
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/api.py and
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/analysis/sparse

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

from __future__ import absolute_import, division
import warnings
import pandas as pd
import numpy as np
import sys

from scipy.linalg import eig, lu_factor, lu_solve, solve
import scipy.sparse.linalg
from scipy.sparse import eye
from scipy.sparse.linalg import factorized
import scipy.sparse.csgraph as csgraph
from scipy.sparse import diags, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from six.moves import range
import decimal as D

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt import visualization
from evotpt.analysis_plotting import MonteCarloAnalysis
from evotpt import utils
from evotpt import tpt_analysis
from gpmap import GenotypePhenotypeMap

def transition_path_theory(tm):
    # Transform pandas transition matrix into numpy array. Required for subsequent matrix operations.

    #T = np.array(tm, dtype=float)
    T = tm
    A = [0]
    B = [len(T) - 1]

    """ Convert transition matrix to scipy sparse matrix """
    Ts = scipy.sparse.csr_matrix(T)
    Ts = Ts.tocsr()

    """ Stationary Distribution """
    vals, vecs = scipy.sparse.linalg.eigs(T.transpose(), k=1, which='LR', ncv=None)
    nu = vecs[:, 0].real
    mu = nu / np.sum(nu)

    if np.any(mu < 0):  # still? Then set to 0 and renormalize
                mu = np.maximum(mu, 0.0)
                mu /= mu.sum()

    print("Stat. Distr.:\n", mu)
    """forward committor"""
    qplus = forward_committor(Ts, A, B)
    # # backward committor given that the matrix is reversible.
    qminus = backward_committor(Ts, A, B, mu)
    """netflux"""
    netflux = flux_matrix(Ts, mu, qplus, qminus)

    return netflux, A, B

def forward_committor(Ts, A, B):
    """ Forward Committor """
    X = set(range(Ts.shape[0]))
    A = set(A)
    B = set(B)
    AB = A.intersection(B)
    notAB = X.difference(A).difference(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    L = Ts - eye(Ts.shape[0], Ts.shape[0])

    """Assemble left hand-side W for linear system"""
    """Equation (I)"""
    W = 1.0 * L

    W = W.tocsr()
    """Equation (II)"""
    W = W.todok()
    W[list(A), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(A)), (list(A), list(A))), shape=W.shape).tocsr()

    """Equation (III)"""
    W = W.todok()
    W[list(B), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(B)), (list(B), list(B))), shape=W.shape).tocsr()

    """Assemble right hand side r for linear system"""
    """Equation (I+II)"""
    r = np.zeros(Ts.shape[0])
    """Equation (III)"""
    r[list(B)] = 1.0
    u = spsolve(W, r)

    qplus = u

    return qplus

def backward_committor(Ts, A, B, mu):
    """ Backward Committor """
    X = set(range(Ts.shape[0]))
    A = set(A)
    B = set(B)
    AB = A.intersection(B)
    notAB = X.difference(A).difference(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    pi = mu
    L = Ts - eye(Ts.shape[0], Ts.shape[0])
    D = diags([pi, ], [0, ])

    K = (D.dot(L)).T

    """Assemble left-hand side W for linear system"""
    """Equation (I)"""
    W = 1.0 * L  # K

    """Equation (II)"""
    W = W.todok()
    W[list(A), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(A)), (list(A), list(A))), shape=W.shape).tocsr()

    """Equation (III)"""
    W = W.todok()
    W[list(B), :] = 0.0
    W.tocsr()
    W = W + coo_matrix((np.ones(len(B)), (list(B), list(B))), shape=W.shape).tocsr()

    """Assemble right-hand side r for linear system"""
    """Equation (I)+(III)"""
    r = np.zeros(Ts.shape[0])

    """Equation (II)"""
    r[list(A)] = 1.0

    u = spsolve(W, r)

    qminus = u

    return qminus

def flux_matrix(Ts, mu, qplus, qminus, netflux=True):
    """Flux"""
    D1 = diags((mu * qminus,), (0,))
    D2 = diags((qplus,), (0,))

    flux = D1.dot(Ts.dot(D2))

    """Remove self-fluxes"""
    flux = flux - diags(flux.diagonal(), 0)

    """Return net or gross flux"""
    netflux = flux - flux.T

    """Set negative entries to zero"""
    Q = netflux
    Q = Q.tocoo()

    data = Q.data
    row = Q.row
    col = Q.col

    """Positive entries"""
    pos = data > 0.0

    datap = data[pos]
    rowp = row[pos]
    colp = col[pos]

    Qplus = coo_matrix((datap, (rowp, colp)), shape=Q.shape)

    return Qplus


class PathwayError(Exception):
    """Exception for failed attempt to find pathway in a given flux
    network"""
    pass


def find_bottleneck(F, A, B):
    r"""Find dynamic bottleneck of flux network.
    Parameters
    ----------
    F : scipy.sparse matrix
        The flux network
    A : array_like
        The set of starting states
    B : array_like
        The set of end states
    Returns
    -------
    e : tuple of int
    The edge corresponding to the dynamic bottleneck
    """
    if F.nnz == 0:
        raise PathwayError('no more pathways left: Flux matrix does not contain any positive entries')
    F = F.tocoo()
    n = F.shape[0]

    """Get exdges and corresponding flux values"""
    val = F.data
    row = F.row
    col = F.col

    """Sort edges according to flux"""
    ind = np.argsort(val)
    val = val[ind]
    row = row[ind]
    col = col[ind]

    """Check if edge with largest conductivity connects A and B"""
    b = np.array(row[-1], col[-1])
    if has_path(b, A, B):
        return b
    else:
        """Bisection of flux-value array"""
        r = val.size
        l = 0
        N = 0
        while r - l > 1:
            m = np.int(np.floor(0.5 * (r + l)))
            valtmp = val[m:]
            rowtmp = row[m:]
            coltmp = col[m:]
            C = coo_matrix((valtmp, (rowtmp, coltmp)), shape=(n, n))
            """Check if there is a path connecting A and B by
            iterating over all starting nodes in A"""
            if has_connection(C, A, B):
                l = 1 * m
            else:
                r = 1 * m

        E_AB = coo_matrix((val[l + 1:], (row[l + 1:], col[l + 1:])), shape=(n, n))
        b1 = row[l]
        b2 = col[l]
        return b1, b2, E_AB


def has_connection(graph, A, B):
    r"""Check if the given graph contains a path connecting A and B.
    Parameters
    ----------
    graph : scipy.sparse matrix
        Adjacency matrix of the graph
    A : array_like
        The set of starting states
    B : array_like
        The set of end states
    Returns
    -------
    hc : bool
       True if the graph contains a path connecting A and B, otherwise
       False.
    """
    for istart in A:
        nodes = csgraph.breadth_first_order(graph, istart, directed=True, return_predecessors=False)
        if has_path(nodes, A, B):
            return True
    return False


def has_path(nodes, A, B):
    r"""Test if nodes from a breadth_first_order search lead from A to
    B.
    Parameters
    ----------
    nodes : array_like
        Nodes from breadth_first_oder_seatch
    A : array_like
        The set of educt states
    B : array_like
        The set of product states
    Returns
    -------
    has_path : boolean
        True if there exists a path, else False
    """
    x1 = np.intersect1d(nodes, A).size > 0
    x2 = np.intersect1d(nodes, B).size > 0
    return x1 and x2


def pathway(F, A, B):
    r"""Compute the dominant reaction-pathway.
    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    A : array_like
        The set of starting states
    B : array_like
        The set of end states
    Returns
    -------
    w : list
        The dominant reaction-pathway
    """
    if F.nnz == 0:
        raise PathwayError('no more pathways left: Flux matrix does not contain any positive entries')
    b1, b2, F = find_bottleneck(F, A, B)
    if np.any(A == b1):
        wL = [b1, ]
    elif np.any(B == b1):
        raise PathwayError(("Roles of vertices b1 and b2 are switched."
                            "This should never happen for a correct flux network"
                            "obtained from a reversible transition meatrix."))
    else:
        wL = pathway(F, A, [b1, ])
    if np.any(B == b2):
        wR = [b2, ]
    elif np.any(A == b2):
        raise PathwayError(("Roles of vertices b1 and b2 are switched."
                            "This should never happen for a correct flux network"
                            "obtained from a reversible transition meatrix."))
    else:
        wR = pathway(F, [b2, ], B)
    return wL + wR


def capacity(F, path):
    r"""Compute capacity (min. current) of path.
    Paramters
    ---------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    path : list
        Reaction path
    Returns
    -------
    c : float
        Capacity (min. current of path)
    """
    F = F.todok()
    L = len(path)
    currents = np.zeros(L - 1)
    for l in range(L - 1):
        i = path[l]
        j = path[l + 1]
        currents[l] = F[i, j]

    return currents.min()


def remove_path(F, path):
    r"""Remove capacity along a path from flux network.
    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    path : list
        Reaction path
    Returns
    -------
    F : (M, M) scipy.sparse matrix
        The updated flux network
    """
    c = capacity(F, path)
    F = F.todok()
    L = len(path)
    for l in range(L - 1):
        i = path[l]
        j = path[l + 1]
        F[i, j] -= c
    return F


def pathways(F, A, B, fraction=1.0, maxiter=1000, tol=1e-14):
    r"""Decompose flux network into dominant reaction paths.
    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    A : array_like
        The set of starting states
    B : array_like
        The set of end states
    fraction : float, optional
        Fraction of total flux to assemble in pathway decomposition
    maxiter : int, optional
        Maximum number of pathways for decomposition
    tol : float, optional
        Floating point tolerance. The iteration is terminated once the
        relative capacity of all discovered path matches the desired
        fraction within floating point tolerance
    Returns
    -------
    paths : list
        List of dominant reaction pathways
    capacities: list
        List of capacities corresponding to each reactions pathway in paths
    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes.
        Multiscale Model Simul 7: 1192-1219 (2009)
    """
    F, a, b = add_endstates(F, A, B)
    A = [a, ]
    B = [b, ]

    """Total flux"""
    TF = F.tocsr()[A, :].sum()

    """Total capacity fo all previously found reaction paths"""
    CF = 0.0
    niter = 0

    """List of dominant reaction pathways"""
    paths = []
    """List of corresponding capacities"""
    capacities = []

    while True:
        """Find dominant pathway of flux-network"""
        try:
            path = pathway(F, A, B)
        except PathwayError:
            break
        """Compute capacity of current pathway"""
        c = capacity(F, path)
        """Remove artifical end-states"""
        path = path[1:-1]
        """Append to lists"""
        paths.append(np.array(path))
        capacities.append(c)
        """Update capacity of all previously found paths"""
        CF += c
        """Remove capacity along given path from flux-network"""
        F = remove_path(F, path)
        niter += 1
        """Current flux numerically equals fraction * total flux or is
        greater equal than fraction * total flux"""
        if (abs(CF / TF - fraction) <= tol) or (CF / TF >= fraction):
            break
        if niter > maxiter:
            warnings.warn("Maximum number of iterations reached", RuntimeWarning)
    return paths, capacities


def add_endstates(F, A, B):
    r"""Adds artifical end states replacing source and sink sets.
    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    A : array_like
        The set of starting states
    B : array_like
        The set of end states
    Returns
    -------
    F_new : (M+2, M+2) scipy.sparse matrix
        The artifical flux network with extra end states
    a_new : int
        The new single source a_new = M
    b_new : int
        The new single sink b_new = M+1
    """

    """Outgoing currents from A"""
    F = F.tocsr()
    outA = (F[A, :].sum(axis=1)).getA()[:, 0]

    """Incoming currents into B"""
    F = F.tocsc()
    inB = (F[:, B].sum(axis=0)).getA()[0, :]

    F = F.tocoo()
    M = F.shape[0]

    data_old = F.data
    row_old = F.row
    col_old = F.col

    """Add currents from new A=[n,] to all states in A"""
    row1 = np.zeros(outA.shape[0], dtype=np.int)
    row1[:] = M
    col1 = np.array(A)
    data1 = outA

    """Add currents from old B to new B=[n+1,]"""
    row2 = np.array(B)
    col2 = np.zeros(inB.shape[0], dtype=np.int)
    col2[:] = M + 1
    data2 = inB

    """Stack data, row and col arrays"""
    data = np.hstack((data_old, data1, data2))
    row = np.hstack((row_old, row1, row2))
    col = np.hstack((col_old, col1, col2))

    """New netflux matrix"""
    F_new = coo_matrix((data, (row, col)), shape=(M + 2, M + 2))

    return F_new, M, M + 1

def path_to_pmf(paths, capacities):
    pmf = {}
    capac_sum = sum(capacities)
    capacities_norm = [capac / capac_sum for capac in capacities]

    for path, capac in zip(paths, capacities_norm):
        path = [gpm.data.at[gt, "genotypes"] for gt in path]
        pmf[tuple(path)] = capac
    return pmf

def min_sparse(X):
    if len(X.data) == 0:
        return 0
    m = X.data.min()
    return m if X.getnnz() == X.size else min(m, 0)

if __name__ == "__main__":
    # execute only if run as a script
    outfilename = sys.argv[1].split(".")[0].split("/")[-1]
    gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
    tm, ratio_matrix = utils.transition_matrix(gpm,
                                 population_size=100,
                                 minval=0,
                                 mutation_rate=1,
                                 null_steps=False,
                                 reversibility=True)

    flux_matrix, A, B = transition_path_theory(tm)
    # print("FM\n", flux_matrix)s
    paths, capacities = pathways(flux_matrix, A, B)
    path_pmf = path_to_pmf(paths, capacities)
    # print("Paths:\n", path_pmf)

    number_of_paths = tpt_analysis.number_of_paths(path_pmf)

    sinks_ = tpt_analysis.sinks(ratio_matrix)
    sinks = [gpm.genotypes[sink] for sink in sinks_]
    peaks_ = tpt_analysis.peaks(ratio_matrix)
    peaks = [gpm.genotypes[peak] for peak in peaks_]
    chains_ = tpt_analysis.chains(ratio_matrix)
    chains = []
    for chain in chains_:
        chains.append(tuple([gpm.genotypes[node] for node in chain]))

    ap_pmf = tpt_analysis.adaptive_paths(gpm, path_pmf)
    non_ap_pmf = tpt_analysis.non_adaptive_paths(path_pmf, ap_pmf)
    if ap_pmf:
        path_diff = tpt_analysis.path_difference(ap_pmf, fraction=1)
        length_distr_ap = tpt_analysis.length_distr(ap_pmf)
        length_distr_non_ap = tpt_analysis.length_distr(non_ap_pmf)
        non_adpv_flux = tpt_analysis.non_adaptive_flux(path_pmf, ap_pmf)

    flux_map = visualization.GenotypePhenotypeGraph(gpm, paths=ap_pmf, double_paths=True, paths2=non_ap_pmf, peaks=peaks, sinks=sinks, chains=chains)
    flux_map.draw_map_double_paths(figsize=(10, 7), node_size=15, linewidth=13)
    flux_map.draw_chains(figsize=(10, 7), node_size=15, linewidth=13)

    visualization.pathlength_histogram(length_distr_ap, length_distr_non_ap, outfilename)

    visualization.path_divergence(ap_pmf, fraction=1., interval=0.1)
