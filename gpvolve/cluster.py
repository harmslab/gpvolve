from pyemma.msm import PCCA
import networkx as nx
import numpy as np

def pcca(evomsm, c):
    """Runs PCCA++ [1] to compute a metastable decomposition of MSM states.

    Parameters
    ----------
    c : int.
        Desired number of metastable sets.

    Returns
    -------
    Nothing : None.
        The important metastable attributes are set automatically.

    Notes
    -----
    The metastable decomposition is done using the pcca method of the pyemma.msm.MSM class.
    For more details and references: https://github.com/markovmodel/PyEMMA/blob/devel/pyemma/msm/models/msm.py
    """

    # Get transtion matrix.
    T = np.array(nx.attr_matrix(evomsm, edge_attr="fixation_probability", normalized=True)[0])

    # Computer clusters
    P = PCCA(T, c)

    # Dictionary of cluster assignments for each node.
    assignments = {node: cluster for node, cluster in enumerate(P.metastable_assignment)}

    # Dictionary of cluster memberships (tuple of length c) for each node.
    memberships = {node: tuple(probs) for node, probs in enumerate(P.memberships)}

    # Cluster sets.
    cluster_sets = P.metastable_sets

    return cluster_sets, assignments, memberships
