from gpgraph import GenotypePhenotypeGraph
from .utils import add_self_probability
from .cluster import *
import msmtools.analysis as mana

import warnings
import networkx as nx
import numpy as np

class EvoMSM(GenotypePhenotypeGraph):
    def __init__(self, gpm, *args, **kwargs):
        super().__init__(gpm, *args, **kwargs)

        # Add self-looping edges (Not done by default in GenotypePhenotypeGraph)
        self.self_edges = zip(*np.diag_indices(len(self.gpm.data.genotypes)))
        self.add_edges_from(self.self_edges)

        # Give every node an 'index' attribute, so nodes can be labeled with their index.
        nx.set_node_attributes(self, name="index", values={node: node for node in self.nodes})

        # Set transition matrix
        self._transition_matrix = None

        # Set basic markov chain attributes that can't be attached to networkx object.
        self._timescales = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._peaks = None

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
        """Calculate fixation probability along all edges and build transition matrix"""
        # Split all egdes into two tuples, each containing one node of each pair of nodes at the same position.
        nodepairs = list(zip(*self.edges))  # [(1, 4), (5, 8), (10, 25)] -> [(1, 5, 10), (4, 8, 25)]

        # Get fitnesses of all nodes.
        fitness1 = np.array([self.node[node]['fitness'] for node in nodepairs[0]])
        fitness2 = np.array([self.node[node]['fitness'] for node in nodepairs[1]])

        # Probability of a certain site mutating in a certain genotype when all sites have equal mutation probability.
        mutation_prob = np.array([1/len(list(self.neighbors(node))) for node in nodepairs[0]])
        # mutation_prob = np.array(1 / nx.adjacency_matrix(self).sum(axis=0))[0]  # number of neighbors, exclude

        # Compute fixation probabilities and get edge keys.
        probs = mutation_prob * fixation_model(fitness1, fitness2, **params)

        # Set fixation probability for all edges. Values for the self-looping edges are incorrect at this point.
        edges = self.edges.keys()
        nx.set_edge_attributes(self, name="fixation_probability", values=dict(zip(edges, probs)))

        # Calculate transition matrix diagonal, i.e. self-looping probability.
        self.transition_matrix = add_self_probability(nx.attr_matrix(self, edge_attr="fixation_probability")[0])

        # Update edge attributes of self-looping edges with transition matrix diagonal values.
        diag_indices = np.diag_indices(self.transition_matrix.shape[0])
        diag_vals = self.transition_matrix[diag_indices]
        nx.set_edge_attributes(self, name="fixation_probability", values=dict(zip(self.self_edges, diag_vals)))

        # Update class attributes.
        # self.stationary_distribution = mana.stationary_distribution(self.transition_matrix)
        # self.timescales = mana.timescales(self.transition_matrix)
        # self.eigenvalues = mana.eigenvalues(self.transition_matrix)
        # self.eigenvectors = mana.eigenvectors(self.transition_matrix)
        # self.stationary_distribution = mana.stationary_distribution(self.transition_matrix)

    def step_function(self):
        pass

    def peaks(self):
        """Find nodes without neighbors of higher fitness. Equal fitness allowed.

        Parameters
        ----------
        self : EvoMSM object.
            EvoMSM object with transition matrix.

        Returns
        -------
        _peaks : list of sets.
            List of peaks. Each peak is a set and can contain multiple nodes if it's a flat peak of nodes with identical
            fitness.
        """
        if self._peaks:
            return self._peaks
        else:
            peak_list = []
            for node, fitness in enumerate(self.gpm.data.fitnesses):
                # Get neighbors.
                neighbors = list(self.neighbors(node))
                # Remove self.
                neighbors.remove(node)
                # If fitness is higher than or equal to fitness of neighbors, it's a peak.
                if fitness >= max([self.gpm.data.fitnesses[neighbor] for neighbor in neighbors]):
                    peak_list.append(node)

            # Find connected peaks.
            peak_graph = self.subgraph(peak_list)
            peaks = list(nx.connected_components(peak_graph.to_undirected()))
            self._peaks = peaks

            return self._peaks

    def soft_peaks(self, error):
        """Find nodes without neighbors of higher fitness. Equal fitness allowed. Takes into account error, e.g. if
        fitness1 has one neighbor (fitness2) with higher fitness, fitness1 is still considered a peak if
        fitness1 + error is higher than or equal to fitness2 - error.

        Parameters
        ----------
        self : EvoMSM object.
            EvoMSM object with transition matrix.

        error : list
            List with one error value for each fitness. Must be in same order as fitness/phenotypes array.

        Returns
        -------
        peaks : list of sets.
            List of peaks. Each peak is a set and can contain multiple nodes if it's a flat peak of nodes with identical
            fitness or nodes with indistinguishable fitness within the margin of error.
        """
        peak_list = []
        fitnesses = pow(self.gpm.data.fitnesses, 10)
        error = pow(error, 10)
        floor_fitnesses = fitnesses - error
        for node, fitness in enumerate(fitnesses):
            # Get neighbors.
            neighbors = list(self.neighbors(node))
            # Remove self.
            neighbors.remove(node)
            # If fitness is higher than or equal to fitness of neighbors, it's a peak.
            if fitness + error[node] >= max([floor_fitnesses[neighbor] for neighbor in neighbors]):
                peak_list.append(node)

        # Find connected peaks.
        peak_graph = self.subgraph(peak_list)
        peaks = list(nx.connected_components(peak_graph.to_undirected()))

        return peaks


    @property
    def transition_matrix(self):
        if self._transition_matrix.any():
            return self._transition_matrix
        else:
            try:
                self._transition_matrix = np.array(nx.attr_matrix(self, edge_attr="fixation_probability", normalized=False)[0])
            except KeyError:
                print("Transition matrix doesn't exit yet. Add fixation probabilities first.")

    @transition_matrix.setter
    def transition_matrix(self, T):
        # Check transition matrix.
        if mana.is_transition_matrix(T):
            if not mana.is_reversible(T):
                warnings.warn("The transition matrix is not reversible.")
            if not mana.is_connected(T):
                warnings.warn("The transition matrix is not connected.")
        else:
            warnings.warn("Not a transition matrix. Has to be square and rows must sum to one.")

        self._transition_matrix = T

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

    @clusters.setter
    def clusters(self, clusters):
        if isinstance(clusters, list):
            self._clusters = clusters

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

    # def peaks_(self):
    #     """
    #     A node is defined as peak if it has no neighbor (hamming_distance=1) with a higher fitness (identical
    #     fitnesses are accepted).
    #
    #     """
    #     ratio_matrix = np.nan_to_num(np.outer(np.array(self.gpm.phenotypes), 1 / np.array(self.gpm.phenotypes)))
    #
    #     # Set diagonal to 0.
    #     np.fill_diagonal(ratio_matrix, 0)
    #
    #     # Get adjaceny matrix
    #     A = nx.adjacency_matrix(self)
    #     Ad = A.todense()
    #     np.fill_diagonal((Ad), 0)
    #
    #     # Set non-neighbor entries zero by multiplying with adjacency matrix
    #     ratio_matrix = np.multiply(ratio_matrix, Ad)
    #
    #     # Set ratios above 1 to zero, i.e. discard downhill moves, only keep uphill moves.
    #     ratio_matrix[ratio_matrix >= 1] = 0
    #
    #     # Sum rows and find rows with sum 0. Those rows don't have uphill moves, hence they are peaks.
    #     peak_list = np.where(ratio_matrix.sum(axis=1) == 0)[0]
    #     print(peak_list)
    #
    #     peak_graph = self.subgraph(peak_list)
    #
    #     peaks = list(nx.connected_components(peak_graph.to_undirected()))
    #
    #     return peaks



