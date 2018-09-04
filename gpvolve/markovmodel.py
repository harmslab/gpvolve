from gpgraph import GenotypePhenotypeGraph
from .utils import self_probability
from .cluster import *
import msmtools.analysis as mana

import warnings
import networkx as nx
import numpy as np

class EvoMSM(GenotypePhenotypeGraph):
    def __init__(self, gpm, *args, **kwargs):
        super().__init__(gpm, *args, **kwargs)

        # Set basic markov chain attributes that can't be attached to networkx object.
        self._timescales = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._clusters = None
        self._total_flux = None

    def apply_selection(self, fitness_function, **params):
        """Compute fitness values from a user-defined phenotype-fitness function. A few basic functions can be found in
           gpsolve.fitness. For a direct mapping of phenotype to fitness, use one_to_one without additional parameters.

        Parameters
        ----------
        fitness_function: function.
            A python function that takes phenotypes and additional parameters(optional) and
            returns a list of fitnesses(type=float).

        Returns
        -------
        Nothing: None
            The computed fitness values are automatically stored under self.gpm.data.fitnesses.
        """
        # Add fitnesses column to gpm.data pandas data frame.
        self.gpm.data['fitnesses'] = fitness_function(self.gpm.data.phenotypes, **params)

        # Add node attribute.
        values = {node: fitness for node, fitness in enumerate(self.gpm.data.fitnesses.tolist())}
        nx.set_node_attributes(self, name='fitness', values=values)

    def add_fixation_probability(self, fixation_model, **params):
        # Split all egdes into two tuples, each containing one node of each pair of nodes at the same position.
        nodepairs = list(zip(*self.edges))  # [(1, 4), (5, 8), (10, 25)] -> [(1, 5, 10), (4, 8, 25)]

        # Get fitnesses of all nodes.
        fitness1 = np.array([self.node[node]['fitness'] for node in nodepairs[0]])
        fitness2 = np.array([self.node[node]['fitness'] for node in nodepairs[1]])

        # Compute fixation probabilities and get edge keys.
        probs = fixation_model(fitness1, fitness2, **params)

        # Set edge attribute.
        edges = self.edges.keys()
        nx.set_edge_attributes(self, name="fixation_probability", values=dict(zip(edges, probs)))

        # Add self-probability, i.e. the diagonal of the transition matrix.
        self_probability(self)

        # Check transition matrix.
        if not mana.is_reversible(self.transition_matrix):
            warnings.warn("The transition matrix is not reversible.")
        if not mana.is_connected(self.transition_matrix):
            warnings.warn("The transition matrix is not connected.")

        # Update class attributes.
        T = self.transition_matrix
        self.stationary_distribution = mana.stationary_distribution(T)
        self.timescales = mana.timescales(T)
        self.eigenvalues = mana.eigenvalues(T)
        self.eigenvectors = mana.eigenvectors(T)


    def cluster(self, method, **params):
        cluster_sets, assignments, memberships = method(self, **params)

        if assignments:
            nx.set_node_attributes(self, name="cluster", values=assignments)

        if memberships:
            nx.set_node_attributes(self, name="memberships", values=memberships)

        if cluster_sets:
            self._clusters = cluster_sets

    def flux(self, method, source, target, **params):
        self.source = source
        self.target = target
        net_flux, total_flux, f_comm, b_comm = method(self.transition_matrix, self.source, self.target, **params)

        # Set attributes.
        nx.set_edge_attributes(self, name="flux", values={edge: net_flux[edge[0], edge[1]] for edge in self.edges})
        nx.set_node_attributes(self, name="forward_committor", values={node: qf for node, qf in enumerate(f_comm)})
        nx.set_node_attributes(self, name="backward_committor", values={node: qb for node, qb in enumerate(b_comm)})
        self._total_flux = total_flux

    @property
    def transition_matrix(self):
        try:
            return np.array(nx.attr_matrix(self, edge_attr="fixation_probability", normalized=True)[0])
        except KeyError:
            print("Transition matrix doesn't exit yet. Add fixation probabilities first.")

    @property
    def stationary_distribution(self):
        """The stationary distribution of the genotype-phenotype-map."""
        stat_dist = nx.get_node_attributes(self, name="stationary_distribution")
        if stat_dist:
            return stat_dist
        else:
            stat_dist = {node: prob for node, prob in enumerate(mana.stationary_distribution(self.transition_matrix))}
            print(stat_dist)
            nx.set_node_attributes(self, name="stationary_distribution", values=stat_dist)

            return nx.get_node_attributes(self, name="stationary_distribution")

    @stationary_distribution.setter
    def stationary_distribution(self, stat_dist_raw):
        if isinstance(stat_dist_raw, dict):
            nx.set_node_attributes(self, name="stationary_distribution", values=stat_dist_raw)
        else:
            stat_dist = {node: prob for node, prob in enumerate(stat_dist_raw)}
            nx.set_node_attributes(self, name="stationary_distribution", values=stat_dist)

    @property
    def timescales(self):
        """Get the relaxation timescales corresponding to the eigenvalues in arbitrary units."""
        if self._timescales.any():
            return self._timescales
        else:
            self._timescales = mana.timescales(self.transition_matrix)
            return self._timescales

    @timescales.setter
    def timescales(self, timescales):
        self._timescales = timescales

    @property
    def eigenvalues(self):
        """Get the eigenvalues of the transition matrix"""
        if self._eigenvalues.any():
            return self._eigenvalues
        else:
            self._eigenvalues = mana.eigenvalues(self.transition_matrix)
            return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, eigenvalues):
        self._eigenvalues = eigenvalues

    @property
    def eigenvectors(self):
        """Get the eigenvalues of the transition matrix"""
        if self._eigenvectors.any():
            return self._eigenvectors
        else:
            self._eigenvectors = mana.eigenvectors(self.transition_matrix)
            return self._eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, eigenvectors):
        self._eigenvectors = eigenvectors

    @property
    def clusters(self):
        return self._clusters

    @property
    def cluster_memberships(self):
        return nx.get_node_attributes(self, name="memberships")

    @property
    def cluster_assigments(self):
        return nx.get_node_attributes(self, name="cluster")

    @property
    def source(self):
        """Get source node"""
        return self._source

    @source.setter
    def source(self, source):
        """Set source node/genotype to list of nodes(type=int) or genotypes(type=str)"""
        if isinstance(source, list):
            if not isinstance(source[0], int):
                df = self.gpm.data
                self._source = [df[df['genotypes'] == s].index.tolist()[0] for s in source]
            elif isinstance(source[0], int):
                self._source = source
        else:
            raise Exception("Source has to be a list of at least one genotype(type=str) or node(type=int)")

    @property
    def target(self):
        """Get target node"""
        return self._target

    @target.setter
    def target(self, target):
        """Set target node/genotype to list of nodes(type=int) or genotypes(type=str)"""
        if isinstance(target, list):
            if not isinstance(target[0], int):
                df = self.gpm.data
                self._target = [df[df['genotypes'] == t].index.tolist()[0] for t in target]
            elif isinstance(target[0], int):
                self._target = target
        else:
            raise Exception("Target has to be a list of at least one genotype(type=str) or node(type=int)")

    @property
    def total_flux(self):
        return self._total_flux



