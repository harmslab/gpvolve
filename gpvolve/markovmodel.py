from gpgraph import GenotypePhenotypeGraph
from gpgraph.models import *
import networkx as nx
import pyemma.msm as msm
from .utils import *


class EvoMarkovStateModel(object):
    def __init__(self,
                 gpm,
                 model=moran,
                 selection_gradient=1,
                 population_size=2,
                 two_step=False,
                 source=None,
                 target=None,
                 ):

        # Set parameters
        self.gpm = gpm
        self.model = model
        self.population_size = population_size
        self.two_step = two_step
        self.source = source
        self.target = target
        # Set pcca attributes None
        self._metastable_memberships = None
        self._metastable_assignment = None
        self._metastable_sets = None

        # Save initial phenotype values. Check if phenotypes_ already exists to prevent overwriting.
        if not hasattr(self.gpm.data, 'phenotypes_'):
            self.gpm.data['phenotypes_'] = self.gpm.data.phenotypes

        # Apply selection to phenotypes in gpm
        self.apply_selection(selection_gradient)

        # Initialize GenotypePhenotypeGraph object.
        self.network = GenotypePhenotypeGraph(self.gpm)
        # Build transition matrix.
        self.build_tm()
        # Initialize markov model.
        self.M = msm.markov_model(self.tm)

    def apply_selection(self, selection_gradient):
        """
        Simulate varying selection pressure.
        The selection gradient defines the slope of the phenotype-to-fitness function
        """
        b = 1/selection_gradient - 1
        fitness = [ph_ + b for ph_ in self.gpm.data.phenotypes_]
        max_fit = max(fitness)
        # Normalize to 1
        norm_fitness = [fit / max_fit for fit in fitness]
        # Can't use 'fitness' because add_model() will use 'phenotype'.
        self.gpm.data.phenotypes = norm_fitness

    def build_tm(self, model=None, population_size=None):
        """
        Build transition matrix with given substitution model and population size.
        Matrix can be recalculated without reinitializing the class by calling build_tm() with desired parameters
        """
        # If no substitution model given, use initial model.
        if not model:
            model = self.model
        # If no population size given, use initial value.
        if not population_size:
            population_size = self.population_size

        if self.two_step:
            pass  # Set edges between genotypes with hamming distances

        # Calculate fixation probability
        self.network.add_model(model=model, population_size=population_size)
        # Set probability of acquiring no mutation / Matrix diagonal.
        self.network = self.self_probability(self.network)
        # Create transition matrix.
        tm = nx.attr_matrix(self.network, edge_attr='prob', normalized=True)[0]
        tm = np.array(tm)
        self._tm = tm

    def pcca(self, c):
        pcca = self.M.pcca(c)
        self._metastable_sets = clusters_to_dict(pcca.metastable_sets)
        self._metastable_memberships = pcca.memberships
        self._metastable_assignment = pcca.metastable_assignment

    def committors(self):
        if not self.source or not self.target:
            raise Exception("Define source(type=list) and target(type=list) first.\nEXAMPLE: M.source = ['AYK']")
        else:
            # Get forward and backward committor values
            forward_committor = {i:f_comm for i, f_comm in enumerate(self.M.committor_forward)}
            backward_committor = {i: f_comm for i, f_comm in enumerate(self.M.committor_backward)}
            # Set node attribute
            nx.set_node_attributes(self.network, name="forward_committor", values=forward_committor)
            nx.set_node_attributes(self.network, name="backward_committor", values=backward_committor)

    @property
    def tm(self):
        # If tm was already built, return it.
        if self._tm is not None:
            return self._tm
        # Else, build it from scratch.
        else:
            # Build matrix with initial parameters.
            self.build_tm(self.model, self.population_size)
            return self._tm

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
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
        return self._target

    @target.setter
    def target(self, target):
        if isinstance(target, list):
            if not isinstance(target[0], int):
                df = self.gpm.data
                self._target = [df[df['genotypes'] == t].index.tolist()[0] for t in target]
            elif isinstance(target[0], int):
                self._target = target
        else:
            raise Exception("Target has to be a list of at least one genotype(type=str) or node(type=int)")

    @property
    def stationary_distribution(self):
        return self.M.stationary_distribution

    @property
    def timescales(self):
        return self.M.timescales

    @property
    def eigenvalues(self):
        return self.M.eigenvalues

    @property
    def forward_committor(self):
        return self.M.committor_forward(self.source, self.target)

    @property
    def backward_committor(self):
        return self.M.committor_backward(self.source, self.target)

    @property
    def metastable_sets(self):
        if self._metastable_sets:
            return self._metastable_sets
        else:
            raise Exception("Perform PCCA before calling metastable_sets")

    @property
    def metastable_memberships(self):
        if self._metastable_memberships.any():
            return self._metastable_memberships
        else:
            raise Exception("Perform PCCA before calling memberships")

    @property
    def metastable_assignment(self):
        if self._metastable_assignment.any():
            return self._metastable_assignment
        else:
            raise Exception("Perform PCCA before calling metastable_assignment")


    @staticmethod
    def self_probability(network):
        """Set self-looping probability, sii = 1 - sum(sij for all j)"""
        selfprob = {}
        for node in network.nodes():
            rowsum = sum([tup[2] for tup in network.out_edges(nbunch=node, data='prob')])
            selfprob[(node, node)] = max(0, 1 - rowsum)
        # Add self-looping edges to network.
        network.add_edges_from([(node, node) for node in network.nodes()])
        nx.set_edge_attributes(network, name='prob', values=selfprob)
        return network
