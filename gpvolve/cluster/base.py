from .utils import *
from ..utils import find_max

import msmtools.analysis as mana
import warnings
import networkx as nx
from scipy.sparse import dok_matrix


class GenotypePhenotypeClusters(object):
    def __init__(self, gpmsm, clusters):
        self.gpmsm = gpmsm

        # General cluster properties.
        self._clusters = clusters
        self._assignments = None
        self._memberships = None
        self.cluster_reps = None

        # Build clustered tm
        self._transition_matrix, self._full_transition_matrix = coarse_grain_transition_matrix(gpmsm.transition_matrix,
                                                                                               clusters)

        # Build networkx DiGraph for clusters from transition matrix and set edge attribute.
        self.network = nx.DiGraph()
        self.network.add_edges_from(dok_matrix(self.transition_matrix).keys())
        nx.set_edge_attributes(self.network, name='transition_probability', values=dok_matrix(self.transition_matrix))
        nx.set_node_attributes(self.network, name='size', values={i: len(cluster) for i, cluster in enumerate(self.clusters)})

        # Get cluster representatives.
        self.cluster_rep()

    def cluster_rep(self, nodes=None):
        """Transfer attributes of one representative node per cluster to the node of the cluster that it represents.

        nodes : list (default=None).
            The ith node of 'nodes' will be the representative of the ith cluster. If None, node with highest fitness in
            each cluster will be used as its representative.
        """
        if nodes:
            self.cluster_reps = nodes
        else:
            # Node with highest fitness becomes representative.
            self.cluster_reps = [find_max(self.gpmsm, cluster) for cluster in self.clusters]

        # Get all attributes of the representative nodes
        values = {i: self.gpmsm.node[node] for i, node in enumerate(self.cluster_reps)}
        nx.set_node_attributes(self.network, values)

    @property
    def clusters(self):
        return self._clusters

    @property
    def assignments(self):
        if self._assignments:
            return self._assignments
        elif self.clusters:
            self._assignments = clusters_to_assignments(self.clusters)
            return self._assignments
        elif self._memberships:
            self._assignments = cluster_assignments(self._memberships)
            return self._assignments
        else:
            raise Exception("No cluster assignments defined and no clusters or membership matrix provided.")

    @assignments.setter
    def assignments(self, assignments):
        self._assignments = assignments

    @property
    def memberships(self):
        if isinstance(self._memberships, np.ndarray):
            return self._memberships

    @memberships.setter
    def memberships(self, M):
        self._memberships = M

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, T):
        # Check transition matrix.
        if mana.is_transition_matrix(T):
            if not mana.is_reversible(T):
                warnings.warn("The transition matrix is not reversible.")
            if not mana.is_connected(T):
                warnings.warn("The transition matrix is not connected.")

            self._transition_matrix = T

        else:
            warnings.warn("Not a transition matrix. Has to be square and rows must sum to one.")

    @property
    def full_transition_matrix(self):
        if isinstance(self._full_transition_matrix, np.ndarray):
            return self._full_transition_matrix
        else:
            raise Exception("No full transition matrix found.")

    @classmethod
    def from_memberships(cls, gpmsm, memberships):
        """Build GenotypePhenotypeClusters object from membership matrix.

        Parameters
        ----------
        gpmsm : GenotypePhenotypeMSM.
            A GenotypePhenotypeMSM object.

        memberships : 2D numpy.ndarray.
            A NxM matrix where N is the total number of nodes in 'gpmsm' and M is the number of clusters.
            The element memberships[n, m] is the membership probability of node n to cluster m. The matrix has to be
            row stochastic.

        Returns
        -------
        self : GenotypePhenotypeClusters object.
            Instance of GenotypePhenotypeClusters with memberships and assignments attribute already attached.
        """
        # Turn memberships into assignments and clusters.
        assignments = cluster_assignments(memberships)
        clusters = cluster_sets(assignments)
        print(cls)

        # Create GenotypePhenotypeClusters object.
        self = cls(gpmsm, clusters)
        # Add memberships and assignment properties.
        self._memberships = memberships
        self.assignments = assignments

        return self

