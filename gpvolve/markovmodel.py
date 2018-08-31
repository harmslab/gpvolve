from gpgraph import GenotypePhenotypeGraph

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
        # Add fitnesses column to gpm.data pandas dataframe.
        self.gpm.data['fitnesses'] = fitness_function(self.gpm.data.phenotypes, **params)

        values = {node: fitness for node, fitness in enumerate(self.gpm.data.fitnesses.tolist())}
        # Add node
        nx.set_node_attributes(self, name='fitness', values=values)

    def add_fixation_probability(self, fixation_model, **params):
        probs = {}
        for edge in self.edges:
            f1 = self.node[edge[0]]['fitness']
            f2 = self.node[edge[0]]['fitness']

            probs[edge] = fixation_model(f1, f2, **params)

        nx.set_edge_attributes(self, name='fixation_probability', values=probs)


