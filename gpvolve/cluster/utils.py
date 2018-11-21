import numpy as np
import itertools


def split_clusters(clusters, new):
    """Take an existing set of clusters and add the clusters in 'new' which will effectively split all clusters that
    contain nodes that are defined 'new' into an old and a new cluster. The number and type of nodes will remain the
    same, but their clustering changes. All nodes in 'new' have to exist in 'clusters'.
    Example: clusters = [[1,2,3], [4,5,6,7]], new = [[1], [6,7]] --> [[1], [2,3], [4,5], [6,7]]. The clusters from new
    have been added to clusters for which the first and last cluster of 'clusters' had to be split.
    """
    pass

def coarse_grain_transition_matrix(T, clusters):
    """Coarse-grain a transition matrix based on clusters.

    Parameters
    ----------
    T : 2D numpy.ndarray.
        Full NxN transition matrix that defines transition probabilities between all N nodes.

    clusters : list.
        List of clusters (list of lists). Must contain all nodes from 0 to N, where N is the number of nodes defined in
        'T'.

    Returns
    -------
    cluster_matrix : 2D numpy.ndarray.
        A MxM matrix where M is the number of clusters. The element [i,j] corresponds to the sum of all transition
        probabilities between all nodes in the ith cluster and all nodes in the jth cluster and normalize by the number
        of nodes in the ith cluster -> Average probability of jumping form cluster i to cluster j. 'cluster_matrix' has
        to be row stochastic.

    sorted_T : 2D numpy.ndarray.
        The sorted version of the full transition matrix (sorted according to clusters).
    """
    # Sort T so that rows and columns belonging to the same cluster are next to each other.
    node_order = list(itertools.chain(*clusters))
    sorted_T = T[:, node_order][node_order]

    # Get lengths of all clusters
    lengths = [len(cl) for cl in clusters]

    # Defines the slices of the sorted transition matrix that belong to the same cluster.
    slices = [(0, lengths[0])]
    for length in lengths[1:]:
        prev = slices[-1][1]
        slices.append((prev, prev + length))

    cluster_matrix = np.empty((len(clusters), len(clusters)))

    # Loop over row chunks.
    for i, row in enumerate(slices):
        # Loop over column chunks for each row.
        for j, col in enumerate(slices):
            # Sum all transition probabilities of all nodes in cluster i to all nodes in cluster j
            # and divide by number of nodes in cluster i -> Average probability of moving from cluster i to cluster j.
            cluster_matrix[i, j] = np.sum(sorted_T[row[0]:row[1], col[0]:col[1]]) / lengths[i]

    return cluster_matrix, sorted_T


def clusters_to_assignments(clusters):
    """Turn a list of clusters into a list of cluster assignments.

    Parameters
    ----------
    clusters : list.
        List of clusters. Every cluster is a list of nodes (dtype=int).

    Returns
    -------
    assignments : list
        List of length N where N is the number if total number of nodes the ith element is the number of the cluster to
        which node i belongs.
    """
    assignments = []
    for i, cluster in enumerate(clusters):
        for node in cluster:
            assignments.append(i)
    return assignments


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


def cluster_dist(clst1, clst2, reorder=False):
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

    if reorder:
        # Reorder the clusters of clst2 based on their similarity with clst1.
        clst2_ord = clst2.copy()
        # For each cluster in clst1, find the cluster in clst2 witht the smalles distance.
        order = np.argmin(d_matrix, axis=1)
        # Apply that order to copy of clst2.
        # (There's probably a faster one-liner but I can't find a way that works for sets with diff. number of clusters)
        for i, o in enumerate(order):
            # Remove item with index o from clst2_ord
            clst2_ord.pop(o)
            # Insert item with index o from clst2.
            clst2_ord.insert(i, clst2[o])

        return d_matrix, [clst1, clst2_ord]

    return d_matrix


def metastability(transition_matrix, clusters):
    """Calculate the metastability of a transition matrix."""
    T = transition_matrix.copy()

    # Reorder T, so that rows and columns belonging to the same cluster are next to each other.
    cluster_diag_order = list(itertools.chain(*clusters))
    S = T[:, cluster_diag_order][cluster_diag_order]

    trace = 0
    start = 0
    for cluster in clusters:
        end = start + len(cluster)
        # The probability of transitioning within a cluster normalized by the total probability of
        # leaving or staying in a certain cluster, which is equal to the length of that cluster, since rows sum to 1.
        trace += np.sum(S[start:end, start:end]) / len(cluster)

        start = end

    metastabi = trace / len(clusters)
    return metastabi


def crispness(membership_matrix, clusters):
    """Calculate the crispness of clustering."""
    # Reorder M, so that rows belonging to the same cluster are next to each other.
    M = membership_matrix.copy()
    cluster_order = list(itertools.chain(*clusters))
    S = M[cluster_order,:]

    trace = 0
    start = 0
    for i, cluster in enumerate(clusters):
        end = start + len(cluster)
        # The probability of transitioning within a cluster normalized by the total probability of
        # leaving or staying in a certain cluster, which is equal to the length of that cluster, since rows sum to 1.
        trace += np.sum(S[start:end, i]) / len(cluster)

        start = end

    crisp = trace / len(clusters)
    return crisp