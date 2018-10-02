from msmtools.flux import pathways as pw
from gpmap.utils import hamming_distance
from .utils import combinations, path_prob

import networkx as nx


def flux_decomp(flux_matrix, source, target, fraction=1, maxiter=1000):
    paths, capacities = pw(flux_matrix, source, target, fraction=fraction, maxiter=maxiter)

    pathways = {tuple(path[0]): path[1] for path in zip(paths, capacities)}

    return pathways

def exhaustive_enumeration(msm, source, target):
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


