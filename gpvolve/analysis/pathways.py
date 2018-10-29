from ..utils import monotonic_incr, combinations
from gpmap.utils import hamming_distance
import itertools
from scipy.stats import entropy


def mean_kullback_leibler_dist(sequences):
    """Mean Kullback-Leibler distance of two discrete distributions.
    Sum of pairwise K-L distances of all pairwise combinations of sequences.
    Could be used to get an average distance measure for a set of evol. path probabilities.

    Parameters
    ----------
    sequences : iterable.
        Any iterable that holds at least two iterables of numerical values.

    Returns
    -------
    mean_KL : float.
        Mean Kullback-Leibler Distance.
    """
    pairs = itertools.combinations(sequences, 2)
    KL = 0
    for pair in pairs:
        KL += entropy(pair[0], pair[1])  # If two sequences given it calculates KL-dist. not entropy (--> scipy docs).

    if KL > 0:
        mean_KL = KL / len(list(pairs))
    else:
        mean_KL = KL
    return mean_KL


def mean_path_divergence(G, paths):
    """Calculate the divergence of a paths ensemble according to Lobkovsky, 2011 [1].

    Parameters
    ----------
    G : GenotypePhenotypeGraph object.
        Any GenotypePhenotypeGraph object or objects of classes that inherit from one,
        like GenotypePhenotypeMSM.

    paths : dict.
        Dictionary of paths (keys) and probabilities (values).
        Example: {(0,1,3): 0.9, (0,2,3): 0.1}

    Returns
    -------
    divergence : float.
        A measure of divergence published as equation (2) in [1].

    References
    ----------
    [1] A. E. Lobkovsky, Y. I. Wolf, and E. V. Koonin.
    Predictability of evolutionary trajecto- ries in
    fitness landscapes. PLoS Comput. Biol., 7:e1002302, 2011.
    """

    # Get all possible pairwise combinations of paths.
    ppairs = itertools.combinations(paths, 2)

    divergence = 0

    for ppair in ppairs:
        ppair_hdist = 0
        # Set combined length of pair
        l = len(ppair[0]) + len(ppair[1])

        for i, path in enumerate(ppair):
            # Define other path
            other_path = ppair[abs(i - 1)]
            for node in path:
                # Repeat node, so we can get all combinations of
                # that node with all nodes of the other path.
                a = [node] * len(other_path)
                npairs = zip(a, other_path)
                for npair in npairs:
                    # Get hamming distance
                    ppair_hdist += hamming_distance(G.node[npair[0]]["binary"], G.node[npair[1]]["binary"])

        # Distance between paths.
        ppair_dist = ppair_hdist / l
        # Get both path probabilities.
        path_probs = list(paths.values())
        # Add divergence of this pair to total divergence
        divergence += ppair_dist * path_probs[0] * path_probs[1]

    return divergence



def adaptive_paths(paths, fitnesses):
    adaptive_paths = []
    for path in paths:
        if monotonic_incr(path, fitnesses):
            adaptive_paths.append(path)

    return adaptive_paths


def forward_paths(paths, msm, source, target):
    fp = []

    comb = combinations(source, target)
    min_dist = hamming_distance(msm.gpm.data.binary[source[0]], msm.gpm.data.binary[target[0]])

    for path in paths:
        if len(path) - 1 == min_dist:
            fp.append(path)

    return fp


