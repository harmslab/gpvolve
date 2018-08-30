from gpgraph import GenotypePhenotypeGraph
from gpgraph.models import *
import networkx as nx
import pyemma.msm as msm
from .utils import *
from scipy import sparse


class EvoMarkovStateModel(object):
    """Object for evolutionary markov state models and their analysis.

    Parameters
    ----------
    gpm : genotype-phenotype-map object.
        Contains (at least) the genotypes, phenotypes/fitnesses and wildtype data of the
        genotype-phenotype-map of interest. A genotype-type-map object can be readily created
        using the gpmap API (https://github.com/harmslab/gpmap).

    model : function. (default=moran)
        A function that returns the probability of an arising mutation being fixed in a population.
        Custom models are allowed but they must take at least two arguments, (1) single fitness
        value of the current genotype of the homogeneous population [type=float] and (2) single
        fitness value of the arising mutant [type=float].

    selection_gradient : float, int. (default=1)
        The selection gradient defines the slope of the mapping from phenotype to fitness and
        stores the fitnesses as gpm.data.phenotypes. The original phenotype values
        will be stored as gpm.data.phenotypes_. A selection gradient of 1 means that the raw
        phenotype values will be used as fitnesses. Gradients above 1 increase and gradients < 1
        decrease relative fitness differences.

    population_size : positive int. (default=1)
        The (effective) population size can be used for computing fixation probabilities.

    two_step_probability : float between 0 and 1. (default=0)
        A factor that rescales the fixation probability of double mutants. 0 = no double mutants.

    source : None, list. (default=None)
        A list of at least one genotype(type=str) or node(type=int) that will serve as source
        for the probability flux, i.e. the starting point of all paths.

    target : None, list. (default=None)
        A list of at least one genotype(type=str) or node(type=int) that will serve as target
        for the probability flux, i.e. the ending point of all paths.

    Attributes
    ----------
    stationary_distribution : numpy.ndarray.
        The stationary distribution of the genotype-phenotype-map. Every element i corresponds to
        the probability of finding a population at the ith genotype after an infinite amount of time.

    eigenvalues : numpy.ndarray.
        The eigenvalues of the transition matrix.

    timescales : numpy.ndarray.
        The relaxation timescales corresponding to the eigenvalues in arbitrary units.

    source : list.
        The given source.

    target : list.
        The given target.

    net_flux : numpy.ndarray.
        The reactive flux between all pairs of nodes.

    total_flux : float.
        The total flux between source and target.

    coarse_net_flux : numpy.ndarray.
        The reactive flux between all metastable sets.

    coarse_total_flux : float.
        The total flux between source and target.

    tm : numpy.ndarray.
        The transition matrix. Each element [i,j] corresponds to the probability of a homogeneous population of
        genotype i becoming homogeneous for genotype j, i.e. moving along the edge (i,j).

    forward_committor : numpy.ndarray.
        An array of floats. Each element i corresponds to the probability of a population reaching the target before
        reaching the source when the population is found at genotype i.

    backward_committor : numpy.ndarray.
        An array of floats. Each element i corresponds to the probability of the population having last visited the
        source instead of the target when the population is found at genotype i.

    metastable_sets : list of numpy.ndarrays.
        Each numpy.ndarray corresponds to a metastable set of genotypes. The assignment is based on the metastable
        membership probability.

    metastable_memberships : numpy.ndarray.
        NxC array, representing a row-stochastic matrix, where  N is the number of nodes/genotypes and C is the number
        of metastable sets and. The element j of each row i corresponds the the probability of genotype i being in
        respective cluster j. Each row must sum to 1.

    metastable_assignment : numpy.ndarray.
        Array of length N where N is the number of nodes/genotypes. Each element i contains the cluster that respective
        genotype i belongs to.

    Notes
    -----
    This module strongly relies on the following modules: pyemma.msm [1] and msmtools [2], gpgraph [3] and networkx [4].
    This module can be seen as a wrapper that sets up a genotype-phenotyp-map for the analyses that are subsequently
    carried out by pyemma.msm and msmtools and we therefore strongly recommend to read the docs and references of those
    two modules.

    References
    ----------
    [1] https://github.com/markovmodel/PyEMMA
    [2] https://github.com/markovmodel/msmtools
    [3] https://github.com/Zsailer/gpgraph/blob/master/gpgraph/base.py
    [4] https://github.com/networkx/networkx

    """

    def __init__(self,
                 gpm,
                 model=moran,
                 selection_gradient=1,
                 population_size=1,
                 two_step_probability=0,
                 source=None,
                 target=None,
                 ):

        # Set parameters
        self.gpm = gpm
        self.model = model
        self.population_size = population_size
        self.two_step_prob = two_step_probability
        self.source = source
        self.target = target
        # Set pcca attributes to None
        self._metastable_memberships = None
        self._metastable_assignment = None
        self._metastable_sets = None
        # Set tpt to None
        self._tpt = None
        self._tpt_coarse = None

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
        """Transforms the phenotype data to match the given selection gradient and stores them as gpm.data.phenotypes.

        Parameters
        ----------
        selection_gradient : float, int. (default=1)
            The selection gradient defines the slope of the phenotype fitness function.

        Returns
        -------
        Nothing: None.
            Fitnesses are automatically stored in gpm.data.phenotypes.
        """
        b = 1/selection_gradient - 1
        fitness = [ph_ + b for ph_ in self.gpm.data.phenotypes_]
        max_fit = max(fitness)
        # Normalize to 1
        norm_fitness = [fit / max_fit for fit in fitness]
        # Can't use 'fitness' because add_model() will use 'phenotype'.
        self.gpm.data.phenotypes = norm_fitness

    def build_tm(self, model=None, population_size=None, two_step_probability=None):
        """Build transition matrix with given population genetics parameters and set tm attribute. Matrix can be
        recalculated without reinstantiating the class by calling build_tm() with desired parameters.

        Parameters
        ----------
        model : function. (default=None)
            A function that returns the probability of an arising mutation being fixed in a population.

        population_size : int. (default=None)
            The (effective) population size can be used for computing fixation probabilities.

        two_step_probability : float between 0 and 1. (default=None)
            A factor that rescales the fixation probability of double mutants. 0 = no double mutants.

        Returns
        -------
        Nothing: None.
            Class attribute tm is updated automatically.
        """
        # If tm is recalculated after class init., check if new parameters were given, else use initial parameters.
        if not model:
            model = self.model
        if not population_size:
            population_size = self.population_size
        if not two_step_probability:
            two_step_probability = self.two_step_prob

        # Calculate fixation probability
        add_probability(self.network, edges=self.network.edges(), model=model, population_size=population_size)

        if two_step_probability > 0:
            # Get edges between nodes with a hamming distance of 2
            two_step_edges = self.get_two_step_edges(self.network)
            # Add edges to network
            self.network.add_edges_from(two_step_edges)
            # Calculate fixation probability for the new edges.
            add_probability(self.network,
                            edges=two_step_edges,
                            edge_weight=two_step_probability,
                            model=model,
                            population_size=population_size)

        # Set probability of acquiring no mutation / Matrix diagonal.
        self.network = self.self_probability(self.network)
        # Create transition matrix.
        tm = np.array(nx.attr_matrix(self.network, edge_attr='prob', normalized=True)[0])
        self._tm = tm

    def pcca(self, c):
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
        pcca = self.M.pcca(c)
        self._metastable_sets = pcca.metastable_sets
        #self._metastable_sets = clusters_to_dict(pcca.metastable_sets)
        self._metastable_memberships = pcca.memberships
        self._metastable_assignment = pcca.metastable_assignment

    def committors(self):
        """Assign each node a forward and backward committor value"""
        if not self.source or not self.target:
            raise Exception("Define source(type=list) and target(type=list) first.\nEXAMPLE: M.source = ['AYK']")
        else:
            # Get forward and backward committor values
            forward_committor = {i:f_comm for i, f_comm in enumerate(self.forward_committor)}
            backward_committor = {i: f_comm for i, f_comm in enumerate(self.backward_committor)}
            # Set node attribute
            nx.set_node_attributes(self.network, name="forward_committor", values=forward_committor)
            nx.set_node_attributes(self.network, name="backward_committor", values=backward_committor)

    def tpt(self):
        """Compute the reactive flux from source to target and store the reactive flux object as class attribute.

        Notes
        ----------
        The reactive flux object is obtained from the pyemma.msm.tpt function.
        For more details and references: https://github.com/markovmodel/PyEMMA/blob/devel/pyemma/msm/api.py
        """
        self._tpt = msm.tpt(self.M, self.source, self.target)

    def tpt_coarse(self, clusters):
        """Coarse-grains the flux onto user-defined sets and store coarse-grained tpt object as class attribute.

        Notes
        ----------
        coarse_grain is a function of the msmtools.flux.ReactiveFlux class.
        For more details and references: https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/reactive_flux.py
        """
        if self._tpt:
            self._tpt_coarse = self._tpt.coarse_grain(clusters)[1]
        else:
            self._tpt = msm.tpt(self.M, self.source, self.target)
            self._tpt_coarse = self._tpt.coarse_grain(clusters)[1]

    def pathways(self, coarse=False, fraction=1., maxiter=1000):
        """Decompose flux into dominant pathways.

        Parameters
        ----------
        coarse : Boolean. (default=False)
            If True: Compute pathways along metastable sets. If False: Compute pathways along individual states.

        fraction: float. (default=1.0)
             Fraction of total flux to assemble in pathway decomposition.

        maxiter : int. (default=1000)
            Maximum number of pathways for decomposition.

        Returns
        -------
        paths : list.
            List of dominant reaction pathways.

        capacities : list.
             List of capacities corresponding to each reactions pathway in paths

        Notes
        -----
        coarse_grain is a function of the msmtools.flux.ReactiveFlux class.
        For more details and references: https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/reactive_flux.py
        """
        if coarse:
            return self._tpt_coarse.pathways(fraction=fraction, maxiter=maxiter)
        else:
            return self._tpt.pathways(fraction=fraction, maxiter=maxiter)

    @property
    def net_flux(self):
        """Get net flux between all pairs of nodes"""
        if self._tpt:
            return self._tpt.net_flux
        else:
            raise Exception("Perform tpt before calling net_flux")

    @property
    def total_flux(self):
        """Get toal flux between source and target"""
        if self._tpt:
            return self._tpt.total_flux
        else:
            raise Exception("Perform tpt before calling total_flux")

    @property
    def coarse_net_flux(self):
        """Get net flux between all pairs of metastable sets"""
        if self._tpt_coarse:
            return self._tpt_coarse.net_flux
        else:
            raise Exception("Coarse-grain before calling total_flux")

    @property
    def coarse_total_flux(self):
        """Get total_flux between source and target"""
        if self._tpt_coarse:
            return self._tpt_coarse.total_flux
        else:
            raise Exception("Coarse-grain before calling total_flux")

    @property
    def tm(self):
        """Get the transition matrix. Compute it first if necessary."""
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
    def stationary_distribution(self):
        """The stationary distribution of the genotype-phenotype-map."""
        return self.M.stationary_distribution

    @property
    def timescales(self):
        """Get the relaxation timescales corresponding to the eigenvalues in arbitrary units."""
        return self.M.timescales()

    @property
    def eigenvalues(self):
        """Get the eigenvalues of the transition matrix"""
        return self.M.eigenvalues()

    @property
    def forward_committor(self):
        """Get the forward committor values for each node"""
        return self.M.committor_forward(self.source, self.target)

    @property
    def backward_committor(self):
        """Get the forward committor values for each node"""
        return self.M.committor_backward(self.source, self.target)

    @property
    def metastable_sets(self):
        """Get the metastable sets as computed by self.pcca"""
        if self._metastable_sets:
            return self._metastable_sets
        else:
            raise Exception("Perform PCCA before calling metastable_sets")

    @property
    def metastable_memberships(self):
        """Get the metastable memberships as computed by self.pcca"""
        if self._metastable_memberships.any():
            return self._metastable_memberships
        else:
            raise Exception("Perform PCCA before calling memberships")

    @property
    def metastable_assignment(self):
        """Get the metastable assignments as computed by self.pcca"""
        if self._metastable_assignment.any():
            return self._metastable_assignment
        else:
            raise Exception("Perform PCCA before calling metastable_assignment")

    @staticmethod
    def get_two_step_edges(network):
        """Compute all edges of hamming distance 2.

        Parameters
        ----------
        network : gpgraph object.
            A gpgraph object contains all genotypes and their properties as a network of nodes and edges.

        Returns
        -------
        two_step_edges: list of tuples.
            All edges that connect two genotypes that are two mutational steps away from each other. Reverse steps
            excluded.
        """
        A = nx.adjacency_matrix(network)
        S = shortest_path(A)
        indices = np.where(S == 2)
        two_step_edges = list(zip(indices[0], indices[1]))
        return two_step_edges

    @staticmethod
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
            rowsum = sum([tup[2] for tup in network.out_edges(nbunch=node, data='prob')])
            selfprob[(node, node)] = max(0, 1 - rowsum)
        # Add self-looping edges to network.
        network.add_edges_from([(node, node) for node in network.nodes()])
        nx.set_edge_attributes(network, name='prob', values=selfprob)
        return network
