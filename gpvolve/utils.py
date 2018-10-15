from scipy.sparse.csgraph import shortest_path
import networkx as nx
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import itertools


def sort_clusters_by_nodes(clusters, nodes):
    """Sort clusters based on nodes. The ith cluster will be the cluster that contains the ith node from nodes

    Parameters
    ----------
    clusters : list.
        List of lists, where each list contains n integers corresponding to the n nodes in that cluster

    nodes : list.
        List of nodes (int). Number of nodes should match number of clusters and each cluster should only contain one
        of the nodes from 'nodes'.

    Returns
    -------
    new_order : list.
        List of clusters (dtype=list) sorted by 'nodes'.
    """
    new_order = []
    for node in nodes:
        for cidx, cluster in enumerate(clusters):
            if node in cluster:
                new_order.append(cluster)
                break
    return new_order


def cluster_dist(clst1, clst2):
    """Calculate pairwise distance between two lists of clusters, i.e. two independent clustering results.

    Parameters
    ----------
    clst1 : list.
        Clusters. List of lists where each list contains n integers corresponding to the n nodes in that cluster.

    clst1 : list.
        Clusters. List of lists where each list contains n integers corresponding to the n nodes in that cluster.

    Returns
    -------
    d_matrix : 2D numpy.ndarray.
        N1 x N2 matrix where N1 and N2 are the number of clusters in 'clst1' and 'clst2'. The element [i,j] corresponds
        to the euclidean distance between the ith cluster of 'clst1' and the jth cluster of 'clst2'.

    clst1 : list.
        Returns unchanged clusters 'clst1'

    clst2_ord : list.
        Returns second clusters reordered based on cluster similiarity with 'clst1', i.e. the ith cluster of 'clst2_ord'
        is the cluster of 'clst2' which has the lowest distance to the ith cluster of 'clst1'.
    """
    d_matrix = np.empty((len(clst1), len(clst2)))
    for idx1, c1 in enumerate(clst1):
        for idx2, c2 in enumerate(clst2):
            # Find the smaller and the larger of the two clusters.
            c_s, c_l = sorted([c1, c2], key=len)
            # How many elements do c_s and c_l have in common
            overlap = sum([1 for i in c_s if i in c_l])
            # How many elements are different, normalized by the total number of elements of the larger cluster.
            # Pretend as if boths clusters have same length. Identical elements have distance of 0, different
            # elements have distance of 1.
            length = len(c_l)
            dist = (length - overlap) / length
            d_matrix[idx1, idx2
            ] = dist

    # Reorder the clusters of clst2 based on their similarity with clst1.
    clst2_ord = clst2.copy()
    # For each cluster in clst1, find the cluster in clst2 witht the smalles distance.
    order = np.argmin(d_matrix, axis=1)
    # Apply that order to copy of clst2.
    # (There's probably a faster one-liner but I can't find a way that works for two sets with diff. number of clusters)
    for i, o in enumerate(order):
        # Remove item with index o from clst2_ord
        clst2_ord.pop(o)
        # Insert item with index o from clst2.
        clst2_ord.insert(i, clst2[o])

    return d_matrix, [clst1, clst2_ord]


def crispness(T, clusters):
    """Calculate the crispness of clustering."""
    # Get block diagonal matrix
    block_diag_order = list(itertools.chain(*clusters))
    print(block_diag_order)
    S = T[:, block_diag_order][block_diag_order]

    trace = 0
    start = 0
    for cluster in clusters:
        end = start + len(cluster)
        # The probability of transitioning within a cluster normalized by the total probability of
        # leaving or staying in a certain cluster, which is equal to the length of that cluster, since rows sum to 1.
        trace += np.sum(S[start:end, start:end]) / len(cluster)

        start = end

    crisp = trace / len(clusters)
    return crisp


def cluster_sets(assignments):
    """Take cluster assignments and return cluster sets"""
    a = np.array(assignments)
    sets = []
    clusters = np.unique(a)
    for cluster in clusters:
        sets.append(np.where(a == cluster)[0])

    return np.array(sets)


def cluster_assignments(memberships):
    """Assign each node to a cluster based on that nodes highest membership value."""
    cl_assign = np.argmax(memberships, axis=1)
    return cl_assign


def dictdict_do_dokmatrix(dictdict):
    """Converts dictionary of dictionaries into matrix.
    Example: {0: {1: 0.4, 2: 0.6}, 1: {0: 0.1, 2: 0.9}, 2: {0: 0.5, 1: 0.5}}
    to numpy.ndarray([[0, 0.4, 0.6], [0.1, 0, 0.9,], [0.5, 0.5, 0])
    """
    row = []
    col = []
    data = []
    for out_dic_key, in_dic in dictdict.items():
        for key, val in in_dic.items():
            row.append(out_dic_key)
            col.append(key)
            data.append(val)
    S = csr_matrix((data, (row, col)))

    return S


def paths_to_endnode(paths):
    """Take dictionary of paths (keys) and their probability (values) and remove everything except the last node of the
    path from the key. Is useful for assigning genotypes to clusters based on their relative probability of reaching a
    certain peak.
    """
    endnode_dict = {}
    for path, prob in paths.items():
        try:
            endnode_dict[path[-1]] += prob
        except KeyError:
            endnode_dict[path[-1]] = prob

    return endnode_dict


def euclidean_distance(prev_pmf, current_pmf):
    """Calc. euclidean distance between two probability mass function (pmf). Both pmf have to have the same order
    but not the same length -> current_pmf is allowed to to be longer.
    """
    euclid_dist = 0
    # Calc. euclidean distance of each path in current pmf to pmf at step current - k (current - previous).
    for val in current_pmf:
        try:
            euclid_dist += abs(current_pmf[val] - prev_pmf[val])
        # If a new path has been sampled since last lag interval, euclidean distance = prob. of that path.
        except KeyError:
            euclid_dist += current_pmf[val]
    return euclid_dist


def check_convergence(pmf_dict, conv_func, **params):
    """Calc. for a ordered seqeuence of probability mass functions.

    Parameters
    ----------
    pmf_dict : dict.
        Dictionary of dictionaries. The outer dict. contains the sequence of probability mass functions (dict.) which is
        checked for convergence. The probability mass functions have to be in the form of an ordered dictionary, where
        the key corresponds to different values of a random variable and the values correspond to the probability
        (between 0 and 1) of the variable to take that value.

    conv_func : Python function.
        Function that is going to be used to assess the similarity of two probability mass functions. Has to take two
        dictionaries (prob. mass func.) as first two arguments

    Returns
    -------
    conv : list.
        List containing the similarity values for all consecutive pairs of probability mass functions.
    """
    intervals = list(pmf_dict.keys())
    conv = []
    for i in range(1, len(intervals)):
        prev_pmf = pmf_dict[intervals[i - 1]]
        curr_pmf = pmf_dict[intervals[i]]
        conv.append(conv_func(prev_pmf, curr_pmf, **params))

    return conv


def rm_self_prob(tm):
    """Remove transition matrix diagonal and renormalize rows to 1"""
    T = tm.copy()
    np.fill_diagonal(T, 0)
    row_sums = T.sum(axis=1)
    T_norm = T / row_sums[:, np.newaxis]
    return T_norm


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