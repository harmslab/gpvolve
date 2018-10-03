from msmtools.flux import pathways as pw
from gpmap.utils import hamming_distance
from .utils import combinations, path_prob, rm_self_prob

import networkx as nx
import numpy as np


def flux_decomp(flux_matrix, source, target, fraction=1, maxiter=1000):
    paths, capacities = pw(flux_matrix, source, target, fraction=fraction, maxiter=maxiter)

    pathways = {tuple(path[0]): path[1] for path in zip(paths, capacities)}

    return pathways

def exhaustive_enumeration(msm, source, target):
    """Calculate the probabiliy of all forward paths between source and target

    Parameters
    ----------
    msm : EvoMSM object.
        EvoMSM class object with a transition matrix.

    source : int.
        Starting node for all paths

    target : list (dtype=int).
        List of nodes at which the paths can end.
    """

    # Get all possible paths between all possible pairs of source and target nodes.
    all_paths = []
    pairs = combinations(source, target)
    for pair in pairs:
        all_paths.extend(list(nx.all_shortest_paths(msm, pair[0], pair[1])))

    # Calculate all path probabilities.
    path_probs = {}
    for path in all_paths:
        path_probs[tuple(path)] = path_prob(path, msm.transition_matrix)

    return path_probs

def greedy(T, source=None):
    """Find the 'greedy' path from source to the nearest peak. Always make step with highest probability.

    Parameters
    ----------
    T : 2D numpy.ndarray
        Transition matrix where element T(ij) should correspond to fixation probability from genotype i to j.
        The highest probability of each row should correspond to the step with highest positive fitness difference.

    source : int.
        Starting node for all paths

    target : list (dtype=int).
        List of nodes at which the paths can end.

    Notes
    -----
    Same can be achieved with one iteration of the gillespie algorithm if the transition matrix contains only one
    nonzero value per row, which corresponds to the transition from genotype i to the neighboring genotype j with the
    highest fitness.
    """

    path = [source]
    steps = 0
    while steps:
        curr = path[-1]
        path.append(np.argmax(T[curr]))
    return path


def gillespie(T, source=None, target=None, n_iter=None, r_seed=None, rm_diag=True):
    """Stochastic path sampling. Probability of making a step equals its fixation probability

    Parameters
    ----------
    T : 2D numpy.ndarray
        Transition matrix where element T(ij) should correspond to fixation probability from genotype i to j.

    source : int.s
        Starting node for all paths

    target : list (dtype=int).
        List of nodes at which the paths can end.

    n_iter: int.
        Maximum number of iterations, i.e. paths that will be generated. Make sure that probability mass function of
        found paths has converged, i.e. increasing the number of iterations should'nt change the outcome significantly
        after convergence.

    r_seed: int.
        Random seed. The result of two executions of this algorithm will be identical if the same random seed is used.

    rm_diag: bool (default=True).
        If True, the probability of making step P(i->i) is set to 0 and all P(i->j) are renormalized to sum to 1.
    """
    if rm_diag:
        # Remove matrix diagonal and renormalize to 1.
        T = rm_self_prob(T)

    # Set random seed for repeatability.
    np.random.seed(seed=r_seed)
    counter = 0
    # Dictionary that counts how often paths appeared (indexed by path).
    paths = {}
    while counter < n_iter:
        counter += 1
        path = [source]
        while path[-1] not in target:
            curr = path[-1]
            # Get random number between 0 and 1
            r = np.random.uniform()
            # Choose next state from probability mass function of current states transition matrix row.
            path.append(np.random.choice(np.nonzero(T[curr])[0], p=T[curr][np.nonzero(T[curr])]))

        try:
            paths[tuple(path)] += 1
        except KeyError:
            paths[tuple(path)] = 1

    return paths
