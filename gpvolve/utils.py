from scipy.sparse.csgraph import shortest_path
import networkx as nx


def self_probability(network):
    """Compute the self-looping probability for all nodes. Corresponds to the diagonal of the transiton matrix.

    Parameters
    ----------
    network : gpgraph object.
        A gpgraph object contains all genotypes and their properties as a network of nodes and edges.

    Returns
    -------
    network : gpgraph object.
        The returned network contains self-looping edges with their respective probability as edge attribute.
    """
    selfprob = {}
    for node in network.nodes():
        rowsum = sum([tup[2] for tup in network.out_edges(nbunch=node, data='fixation_probability')])
        selfprob[(node, node)] = max(0, 1 - rowsum)
    # Add self-looping edges to network.
    network.add_edges_from([(node, node) for node in network.nodes()])
    nx.set_edge_attributes(network, name='fixation_probability', values=selfprob)

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


def cluster_positions(network, clusters, scale=None):
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
        for node in clusters[cluster]:
            dist = sp_matrix[peak][node]
            try:
                temp[int(dist)].append(node)
            except KeyError:
                temp[int(dist)] = []
                temp[int(dist)].append(node)
        shells[cluster] = [temp[i] for i in range(0, max(temp.keys())+1)]
    pos = {}
    for i, cluster in shells.items():
        peak_x = network.node[peaks[i]]["forward_committor"]
        peak_y = network.node[peaks[i]]["phenotypes"]
        pos.update(nx.shell_layout(network, cluster, scale=scale, center=[peak_x, peak_y]))

    return pos


