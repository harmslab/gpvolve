from gpgraph import GenotypePhenotypeGraph
from gpgraph.models import *
import networkx as nx


class MarkovModel(object):
    def __init__(self,
                 gpm,
                 model=moran,
                 selection_gradient=1,
                 population_size=2,
                 two_step=False):

        # Set parameters
        self.gpm = gpm
        self.model = model
        self.population_size = population_size
        self.two_step = two_step
        self._tm = None

        # Save initial phenotype values
        self.gpm.data['_phenotypes'] = list(self.gpm.data.phenotypes)

        # Apply selection to phenotypes in gpm
        self.apply_selection(selection_gradient)

        # Initialize GenotypePhenotypeGraph object and calculate transition matrix
        self.network = GenotypePhenotypeGraph(self.gpm)
        self._tm = self.tm()

    def tm(self):
        if self._tm is not None:
            return self._tm
        else:
            if self.two_step:
                pass  # Set edges between genotypes with hamming distances

            self.network.add_model(model=self.model, population_size=self.population_size)  # Calculate fixation prob.
            self.self_probability()  # Set probability of acquiring no mutation / Matrix diagonal.
            tm = nx.attr_matrix(self.network, edge_attr='prob', normalized=True)  # Create transition matrix.
            return tm

    def apply_selection(self, selection_gradient):
        """
        Simulate varying selection pressure.
        The selection gradient defines the slope of the phenotype-to-fitness function
        """
        b = 1/selection_gradient - 1
        fitness = [ph + b for ph in self.gpm.data._phenotypes]
        max_fit = max(fitness)
        norm_fitness = [fit / max_fit for fit in fitness]  # Normalize to 1
        self.gpm.data.phenotypes = norm_fitness  # Can't use 'fitness' because add_model() will use 'phenotype'.

    def self_probability(self):
        """Set self-looping probability, sii = 1 - sum(sij for all j)"""
        selfprob = {}
        for node in self.network.nodes():
            rowsum = sum([tup[2] for tup in self.network.out_edges(nbunch=node, data='prob')])
            selfprob[(node, node)] = max(0, 1 - rowsum)

        self.network.add_edges_from([(node, node) for node in self.network.nodes()])  # Add self-looping edges to network.
        nx.set_edge_attributes(self.network, name='prob', values=selfprob)
