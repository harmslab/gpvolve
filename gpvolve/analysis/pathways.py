from ..utils import monotonic_incr, combinations
from gpmap.utils import hamming_distance

import networkx as nx


def adaptive_paths(paths, fitnesses):
    adaptive_paths = []
    for path in paths:
        if monotonic_incr(path, fitnesses):
            adaptive_paths.append(path)

    return adaptive_paths


def forward_paths(paths, msm, source, target):
    fp = []

    comb = combinations(source, target)
    min_dist = hamming_distance(gpm.data.binary[source[0]], gpm.data.binary[target[0]])

    for path in paths:
        if len(path) - 1 == min_dist:
            fp.append(path)

    return fp