from scipy.sparse.csgraph import shortest_path
import networkx as nx
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np


def rm_self_prob(tm):
    """Remove transition matrix diagonal and renormalize rows to 1"""
    np.fill_diagonal(tm, 0)
    row_sums = tm.sum(axis=1)
    tm_norm = tm / row_sums[:, np.newaxis]
    return tm_norm


def to_dok_matrix(matrix):
    D = dok_matrix(matrix)
    return D


def paths_prob_to_edges_flux(paths_prob):
    """Chops a list of paths into its edges, and calculate the probability
    of that edge across all paths.

    Parameters
    ----------
    paths: list of tuples
        list of the paths.

    Returns
    -------
    edge_flux: dictionary
        Edge tuples as keys, and probabilities as values.
    """
    edge_flux = {}
    for path, prob in paths_prob.items():

        for i in range(len(path)-1):
            # Get edge
            edge = (path[i], path[i+1])

            # Get path probability to edge.
            if edge in edge_flux:
                edge_flux[edge] += prob

            # Else start at zero
            else:
                edge_flux[edge] = prob

    return edge_flux


def path_prob(path, T):
    prob = 1
    for i in range(len(path)-1):
        prob = prob * T[path[i], path[i+1]]
    return prob


def monotonic_incr(sequence, values):
    """See if a sequence of values is monotonically increasing.

    Parameters
    ----------
    sequence : list, tuple, 1D-array (dtype = int).
        Each element is an index for accessing a value from values.

    values : list, tuple, 1D-array (dtype = float, int).
        Stores all possible values that can occur in a sequence.

    Returns
    -------
    bool: True, False.
        True if sequence is monotonically increasing. Allows for neutral steps. False if not monotonically increasing.
    """

    for i in range(len(sequence)-1):
        if values[sequence[i]] > values[sequence[i+1]]:
            return False
    return True


def combinations(source, target):
    c = []
    for s in source:
        for t in target:
            c.append((s, t))
    return c


def add_self_probability(T):
    """Compute the self-looping probability for all nodes. Corresponds to the diagonal of the transiton matrix.

    Parameters
    ----------
    T : numpy matrix.
        Transition matrix.

    Returns
    -------
    network : gpgraph object.
        The returned network contains self-looping edges with their respective probability as edge attribute.
    """

    if not isinstance(T, np.ndarray):
        raise Exception('Transition matrix must be numpy.ndarray')

    # If matrix, make ndarray. Different behavior for.sum
    if isinstance(T, np.matrixlib.defmatrix.matrix):
        T = np.array(T)

    # Reset diagonals to zero. Necessary in case where transition matrix is recalculated with different parameters.
    np.fill_diagonal(T, 0)
    row_sums = T.sum(axis=1)
    row, col = np.diag_indices(T.shape[0])
    T[row, col] = np.ones(row_sums.shape[0]) - row_sums
    print(T)
    return T




def add_probability(network, edges, model, edge_weight=1, **params):
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]

        phenotype1 = network.node[node1]["phenotypes"]
        phenotype2 = network.node[node2]["phenotypes"]

        network.edges[edge]['prob'] = edge_weight * model(phenotype1, phenotype2, **params)


def cluster_peaks(network, clusters):
    """Get fitness peaks of clusters
    Parameters
    ----------
    network : GenotypePhenotypeGraph object
    clusters: Dictionary
              Keys are cluster numbers. Values are lists of nodes of the respective cluster
    """
    if not isinstance(clusters, dict):
        clusters = clusters_to_dict(clusters)
    cluster_peaks = {}
    for cluster, nodes in clusters.items():
        fitnesses = {node: network.node[node]["phenotypes"] for node in nodes}
        cluster_peaks[cluster] = max(fitnesses, key=fitnesses.get)
    return cluster_peaks


def cluster_centers(M, peaks):
    new = {0: M.source[0]}
    for key, value in peaks.items():
        new[key+1] = value
    new[len(peaks)+1] = M.target[0]
    return new


def clusters_to_dict(clusters):
    """Turn list of arrays from pcca.metastable_sets into dictionary"""
    dic = {i: list(cluster) for i, cluster in enumerate(clusters)}
    return dic


def shortest_path_matrix(network):
    """Matrix A with length of shortest path from i to j at A(ij)"""
    A = nx.adj_matrix(network)
    shortest_path_matrix = shortest_path(A)
    return shortest_path_matrix


def cluster_positions(network, clusters, xaxis, yaxis, scale=None):
    """
    Nested list of shells for each cluster. shells = {cluster: [[nodes of h.-dist.=1],[.. h.-dist=2],[.. h.-dist=3]]}
    shells = {2: [[1,3,4] ,[12,23], [28,34,45]]}
    """
    clusters = clusters_to_dict(clusters)
    peaks = cluster_peaks(network, clusters)
    sp_matrix = shortest_path_matrix(network)
    shells = {}
    for cluster, peak in peaks.items():
        temp = {}
        # Get dist of every node in cluster to cluster peak.
        for node in clusters[cluster]:
            dist = sp_matrix[peak][node]
            try:
                temp[int(dist)].append(node)
            except KeyError:
                temp[int(dist)] = []
                temp[int(dist)].append(node)
        shells[cluster] = []
        for i in range(0, max(temp.keys()) + 1):
            try:
                shells[cluster].append(temp[i])
            except KeyError:
                print("Warning: cluster %s seems to be disconnected. This can happen when the clusters are based on tpt"
                      "sets and the connecting node is either source or target" % cluster)
                pass

    pos = {}
    for i, cluster in shells.items():
        try:
            peak_x = network.node[peaks[i]][xaxis]
        except KeyError:
            raise Exception("Node attribute '%s' is not defined. Set desired node attribute first." % xaxis)
        try:
            peak_y = network.node[peaks[i]][yaxis]
        except KeyError:
            raise Exception("Node attribute '%s' is not defined. Set desired node attribute first." % yaxis)

        pos.update(nx.shell_layout(network, cluster, scale=scale, center=[peak_x, peak_y]))

    return pos


def max_prob_matrix(T, source=None, target=None):
    """Transition matrix that only allows the step with maximum probability"""
    indices = np.argmax(T, axis=1)
    indptr = np.array(range(T.shape[0]+1))
    data = np.ones(T.shape[0])
    M = csr_matrix((data, indices, indptr), shape=T.shape).toarray()
    return M