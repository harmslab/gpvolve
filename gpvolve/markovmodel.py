from gpgraph import GenotypePhenotypeGraph
from .utils import self_probability
from .cluster import *

import networkx as nx
import numpy as np

class EvoMSM(GenotypePhenotypeGraph):
    def __init__(self, gpm, *args, **kwargs):
        super().__init__(gpm, *args, **kwargs)

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

    def cluster(self, method, **params):

        assignments, memberships = method(self, **params)

        nx.set_node_attributes(self, name="cluster", values=assignments)

        return memberships

    def flux(self, method, **params):
        pass

    def get_edge_attr(self, attribute):
        pass

    def get_node_attr(self, attribute):
        pass




