#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach / Major parts copied from:
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/api.py and
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/analysis/sparse

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

from __future__ import absolute_import, division
import sys
import scipy
import numpy as np
import pandas as pd
import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve, solve
from six.moves import range

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt import utils
from gpmap import GenotypePhenotypeMap


def tpt(tm):
    # Transform pandas transition matrix into numpy array. Required for subsequent matrix operations.
    T = np.array(tm, dtype=float)
    A = [0]
    B = [3]
    ### ALL CODE BELOW IS COPIED FROM OR INSPIRED BY https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/api.py
    ### AND https://github.com/markovmodel/msmtools/blob/devel/msmtools/analysis/sparse

    # we can compute the following properties from either dense or sparse T
    # stationary dist
    mu = stationary_distribution(T)

    # forward committor
    qplus = forward_committor(T, A, B)
    # # backward committor given that the matrix is reversible.
    qminus = 1.0 - qplus

    # gross flux
    grossflux = flux_matrix(T, mu, qminus, qplus, netflux=True)
    print(grossflux)

    filename = sys.argv[1].split("/")[-1].split(".")[0]
    np.savetxt('%s_flux.txt' % filename, grossflux)


def stationary_distribution(T):
    val, L = eig(T, left=True, right=False)

    """ Sorted eigenvalues and left and right eigenvectors. """
    perm = np.argsort(val)[::-1]

    val = val[perm]
    L = L[:, perm]
    """ Make sure that stationary distribution is non-negative and l1-normalized """
    nu = np.abs(L[:, 0])
    mu = nu / np.sum(nu)
    return mu

def forward_committor(T, A, B):
    X = set(range(T.shape[0]))
    A = set(A)
    B = set(B)
    AB = A.intersection(B)
    notAB = X.difference(A).difference(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    L = T - np.eye(T.shape[0])  # Generator matrix

    """Assemble left hand-side W for linear system"""
    """Equation (I)"""
    W = 1.0 * L
    """Equation (II)"""
    W[list(A), :] = 0.0
    W[list(A), list(A)] = 1.0
    """Equation (III)"""
    W[list(B), :] = 0.0
    W[list(B), list(B)] = 1.0

    """Assemble right hand side r for linear system"""
    """Equation (I+II)"""
    r = np.zeros(T.shape[0])
    """Equation (III)"""
    r[list(B)] = 1.0

    u = solve(W, r)
    return u

def flux_matrix(T, pi, qminus, qplus, netflux=True):
    r"""Compute the TPT flux network for the reaction A-->B.
    Parameters
    ----------
    T : (M, M) ndarray
        transition matrix
    pi : (M,) ndarray
        Stationary distribution corresponding to T
    qminus : (M,) ndarray
        Backward comittor
    qplus : (M,) ndarray
        Forward committor
    netflux : boolean
        True: net flux matrix will be computed
        False: gross flux matrix will be computed
    Returns
    -------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.
    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT). The central object used in transition path theory is the
    forward and backward comittor function.
    See also
    --------
    committor.forward_committor, committor.backward_committor
    """
    ind = np.diag_indices(T.shape[0])
    flux = pi[:, np.newaxis] * qminus[:, np.newaxis] * T * qplus[np.newaxis, :]
    """Remove self fluxes f_ii"""
    flux[ind] = 0.0
    """Return net or gross flux"""
    if netflux:
        return to_netflux(flux)
    else:
        return flux

def to_netflux(flux):
    r"""Compute the netflux from the gross flux.
        f_ij^{+}=max{0, f_ij-f_ji}
        for all pairs i,j
    Parameters
    ----------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.
    Returns
    -------
    netflux : (M, M) ndarray
        Matrix of netflux values between pairs of states.
    """
    netflux = flux - np.transpose(flux)
    """Set negative fluxes to zero"""
    ind = (netflux < 0.0)
    netflux[ind] = 0.0
    return netflux

gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
tm = utils.transition_matrix(gpm.data,
                             gpm.wildtype,
                             gpm.mutations,
                             population_size=5,
                             mutation_rate=1,
                             null_steps=True,
                             reversibility=True)

tpt(tm)