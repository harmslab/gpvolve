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

        self.gpm = gpm
        self.model = model
        self.population_size = population_size
        self.two_step = two_step

        # Save initial phenotype values
        if not hasattr(self.gpm.data, 'phenotypes_'):  # Check if phenotypes_ already exists to prevent overwriting.
            self.gpm.data['phenotypes_'] = self.gpm.data.phenotypes

        # Apply selection to phenotypes in gpm
        self.apply_selection(selection_gradient)

        # Initialize GenotypePhenotypeGraph object and calculate transition matrix
        self.network = GenotypePhenotypeGraph(self.gpm)
        self.build_tm()


    def build_tm(self, model=None, population_size=None):
        """
        Build transition matrix with given substitution model and population size.
        Matrix can be recalculated without reinitializing the class by calling build_tm() with desired parameters
        """
        if not model:  # If no substitution model given, use initial model.
            model = self.model
        if not population_size:  # If no population size given, use initial value.
            population_size = self.population_size

        if self.two_step:
            pass  # Set edges between genotypes with hamming distances

        self.network.add_model(model=model, population_size=population_size)  # Calculate fixation prob.
        self.self_probability()  # Set probability of acquiring no mutation / Matrix diagonal.
        tm = nx.attr_matrix(self.network, edge_attr='prob', normalized=True)  # Create transition matrix.
        self._tm = tm

    @property
    def tm(self):
        if self._tm is not None:  # If tm was already built, return it.
            return self._tm
        else:  # Else, build it from scratch.
            self.build_tm(self.model, self.population_size)  # Build matrix with class parameters.
            return self._tm


    def apply_selection(self, selection_gradient):
        """
        Simulate varying selection pressure.
        The selection gradient defines the slope of the phenotype-to-fitness function
        """
        b = 1/selection_gradient - 1
        fitness = [ph + b for ph in self.gpm.data.phenotypes_]
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
