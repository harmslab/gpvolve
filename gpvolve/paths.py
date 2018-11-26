from msmtools.flux import pathways as pw
from gpmap.utils import hamming_distance
from .utils import combinations, path_prob, rm_self_prob, add_self_probability, euclidean_distance

import networkx as nx
import numpy as np
import warnings


def flux_decomp(flux_matrix, source, target, fraction=1, maxiter=1000):
    """Decomposition of flux into pathways"""
    paths, capacities = pw(flux_matrix, source, target, fraction=fraction, maxiter=maxiter)

    pathways = {tuple(path[0]): path[1] for path in zip(paths, capacities)}

    return pathways

def exhaustive_enumeration(graph, source, target, edge_attr, normalize=False, rm_diag=False):
    """Calculate the probabiliy of all forward paths between source and target

    Parameters
    ----------
    graph : networkx.DiGraph.
        networkx.DiGraph. Can be build from a transition matrix (numpy matrix/array) using networkx.from_numpy_matrix.

    source : int.
        Starting node for all paths

    target : list (dtype=int).
        List of nodes at which the paths can end.

    edge_attr : str.
        Edge attribute that is used to build transition matrix. Only use 'weight' if it's an explicitly defined edge
        attribute, otherwise networkx will build an adjacency matrix instead of a transition matrix.

    normalize : bool.
        If True, normalize each path probability by the sum of all probabilities, so they sum to 1.

    rm_diag : bool.
        If True, the matrix diagonal, i.e. the probability of self-looping, is set to 0. This will skew the path
        probabilities.

    Returns
    -------
    path_probs : dict.
        Dictionary with paths (dtype=tuple) as dict. keys and probabilities as dict. values.
    """
    # Check arguments.
    if edge_attr == 'weight':
        print("If edge_attr='weight', the transition matrix might be a simple adjacency matrix, unless 'weight' is an explicitly defined edge attribute of your DiGraph.")

    # Get all possible paths between all possible pairs of source and target nodes.
    all_paths = []
    pairs = combinations(source, target)
    for pair in pairs:
        all_paths.extend(list(nx.all_shortest_paths(graph, pair[0], pair[1])))

    # Build transition matrix.
    try:
        P = np.array(nx.attr_matrix(graph, edge_attr)[0])
        T = add_self_probability(P)
    except KeyError:
        raise Exception("Edge attribute %s does not exist. Add edge attribute or choose existing." % edge_attr)

    if rm_diag:
        T = rm_self_prob(T)

    # Calculate all path probabilities.
    path_probs = {}
    for path in all_paths:
        path_probs[tuple(path)] = path_prob(path, T)

    if normalize:
        # Normalize, i.e. divide by the sum of all path probabilities.
        p_sum = sum(path_probs.values())
        for path, prob in path_probs.items():
            path_probs[path] = prob/p_sum

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

    Returns
    -------
    path : list.
        The path from source to a peak (list of integers). Integers correspond to transition matrix indices.

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


def path_sampling(Tm, source=None, target=None, max_iter=None, interval=None, conv_crit=0.01, r_seed=None, out='frac', rm_diag=False, mute=False):
    """Stochastic path sampling. Probability of making a step equals its fixation probability

    Parameters
    ----------
    Tm : 2D numpy.ndarray
        Transition matrix where element T(ij) should correspond to fixation probability from genotype i to j.

    source : int.s
        Starting node for all paths

    target : list (dtype=int).
        List of nodes at which the paths can end.

    max_iter: int.
        Maximum number of iterations, i.e. paths that will be generated. Make sure that probability mass function of
        found paths has converged, i.e. increasing the number of iterations should'nt change the outcome significantly
        after convergence.

    interval: int.
        After each interval the sampling is checked for convergence.

    r_seed: int.
        Random seed. The result of two executions of this algorithm will be identical if the same random seed is used.

    rm_diag: bool (default=True).
        If True, the probability of making step P(i->i) is set to 0 and all P(i->j) are renormalized to sum to 1.

    Returns
    -------
    paths_at_intervals : dict.
        Dictionary of dictionaries. Outer dictionary contains all paths and their counts or probability for each
        interval. Inner dictionary is dictionary of paths (keys, dtype=tuple) and their count or probability.

    """

    ### DEFINITELY NEEDS A MORE SOPHISTICATED WAY OF CHECKING CONVERGENCE. IF EUCL. DIST. IS USED, AT LEAST ACCOUNT
    ### FOR THE NUMBER OF ITERATIONS/ ACCOUNT FOR THE EFFECT THAT ONE SAMPLE CAN MAKE TO THE ALREADY EXISTING NUMBER OF
    ### SAMPLES

    if not isinstance(source, int):
        raise Exception("source must be single node")

    T = Tm.copy()

    if not max_iter:
        max_iter = float('inf')
        if not interval:
            raise Exception("Have to provide either maximum number of iteratons ('max_iter') or 'interval'")

    if rm_diag:
        # Remove matrix diagonal and re-normalize to 1 because we aren't interested in self-looping.
        T = rm_self_prob(T)

    if interval:
        if max_iter % interval > 0:
            raise Exception("The number of iterations ('max_iter') has to be a multiple of 'intervals'")

    # Set random seed for repeatability.
    np.random.seed(seed=r_seed)
    counter = 0
    # Dictionary that contains all paths and their counts for each interval.
    paths_at_intervals = {0: {}}
    # Dictionary that counts how often paths appeared (indexed by path).
    paths = {}

    conv_metric = None
    while counter < max_iter:
        counter += 1
        path = [source]

        while path[-1] not in target:
            state = path[-1]
            if state == source and len(path) > 1:
                break
            # Indices of states with nonzero transition probability.
            nonzero_ind = np.nonzero(T[state])[0]
            # Probability mass function for states with nonzero transition probability.
            probs = T[state][np.nonzero(T[state])]
            # As long as the state has not changed, continue to randomly pick new state from the prob. distribution.
            while state == path[-1]:
                # Choose next state from probability mass function of states with nonzero transition probability.
                state = np.random.choice(nonzero_ind, p=probs)

            path.append(state)

        if state == source and len(path) > 1:
            counter -= 1
            continue

        # Count of sampled path + 1.
        try:
            paths[tuple(path)] += 1
        except KeyError:
            paths[tuple(path)] = 1
        print(paths)

        if counter % interval == 0:
            if out == 'count':
                paths_at_intervals[counter] = paths.copy()

            elif out == 'frac':
                count_sum = sum(paths.values())
                paths_at_intervals[counter] = {path: counts/count_sum for path, counts in paths.items()}

                # Use the euclidean distance between two probability mass functions as convergence proxy.
                conv_metric = euclidean_distance(paths_at_intervals[counter - interval], paths_at_intervals[counter])

                if conv_metric < conv_crit:
                    if not mute:
                        print("Converged after %s iterations. Euclidean distance: %s Convergence criterion: %s" % (counter, conv_metric, conv_crit))
                    return paths_at_intervals[counter], paths_at_intervals

    print("Did not converge after %s iterations. Last eucl. distance: %s Convergence criterion: %s" % (counter, conv_metric, conv_crit))
    return paths_at_intervals[counter], paths_at_intervals
