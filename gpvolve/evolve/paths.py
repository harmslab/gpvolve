import numpy as np

def paths_and_probabilities(T, source, target, *args, **kwargs):
    """Find the most likely shortest path between a source and target.

    Parameters
    ----------
    T : 2d array
        transition matrix
    source : int
        index of source node.
    target : int or list of ints
        index of target node(s).


    Notes
    -----
    ``args`` and ``kwargs`` get passed to the transition_model function.

    Returns
    -------
    paths : list
        a list of lists representing all paths from source to target.
    probabilities : list
        a list of the probabilities for each path between source and target.
    """
    # Build a list of probabilities
    probabilities = list()
    # Iterate through all paths in paths-list
    transition_matrix = T
    for p in paths:
        path_length = len(p)
        # Begin by giving this path a probability of 1.
        pi = 1
        # Iterate through edges and multiply by the
        # transition probability of that edge.
        for i in range(path_length-1):
            pi *= transition_matrix[p[i],p[i+1]]
        # Append pi to probabilities
        probabilities.append(pi)
    # Return normalized probabilities. If sum(probabilities) is zero, return
    # a vector of zeros.
    if sum(probabilities) == 0 :
        return paths, list(probabilities)
    else:
        return paths, list(np.array(probabilities)/sum(probabilities))


def flux(T, source, target, *args, **kwargs):
    """Calculate the probability at each edge, i.e. the flux of probability
    through each edge.

    Parameters
    ----------
    G : GenotypePhenotypeGraph (Subclass of networkx.DiGraph)
        Networkx object constructed for genotype-phenotype map.
    source : int
        index of source node.
    target : int
        index of target node.
    transition_model : callable
        function to compute transition probabilities
    """
    paths, probs = paths_and_probabilities(T, source, target,
        *args,
        **kwargs
    )
    # convert paths to tuples
    paths = [tuple(p) for p in paths]
    # map path to probability
    traj = dict(zip(paths, probs))
    # flux mapping dictionary (edge to probability)
    flux = OrderedDict([(edge,0) for edge in G.edges()])
    for path, prob in traj.items():
        # walk through trajectory and add probabilities to each edge.
        for last_i, node in enumerate(path[1:]):
            i = path[last_i]
            j = node
            flux[(i,j)] += prob
    # Return edge probabilities as array.
    return np.array(list(flux.values()))
