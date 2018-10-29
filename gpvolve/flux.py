from msmtools.flux import tpt
import networkx as nx
from scipy.sparse import dok_matrix

def paths_prob_to_edges_flux(paths_prob):
    """Chops a list of paths into its edges, and calculate the probability
    of that edge across all paths.

    Parameters
    ----------
    paths: list of tuples
        list of the paths.

    Returns
    -------
    edge_flux: dictionary
        Edge tuples as keys, and probabilities as values.
    """
    edge_flux = {}
    for path, prob in paths_prob.items():
        edges = []
        for i in range(len(path)-1):
            # Get edge
            edges.append((path[i], path[i+1]))

        # If paths loop over same pair of states multiple times per path, the probability shouldnt be summed.
        uniq_edges = set(edges)

        for edge in uniq_edges:
            # Get path probability to edge.
            if edge in edge_flux:
                edge_flux[edge] += prob

            # Else start at zero
            else:
                edge_flux[edge] = prob

    return edge_flux


class TransitionPathTheory(object):
    """Class for calculating reactive flux of a markov state model.

    Parameters
    ----------
    EvoMSM : EvoMSM object.
        EvoMSM object with an ergodic transition matrix.

    source : list.
        List of nodes that will be treated as flux source.

    target : list.
        List of nodes that will be treated as flux sink.

    Attributes
    ----------
    net_flux : 2D numpy.ndarray.
        Net flux between all pairs of genotypes/nodes.

    total_flux : float.
        The total probability flux that leaves source and ends at sink without passing through source again.

    forward_committor: 1D numpy.ndarray.
        Forward committor values for all nodes.

    backward_committor: 1D numpy.ndarray.
        Backward committor values for all nodes.

    Notes
    -----
    This class can be seen as a wrapper for the msmtools function tpt. Please read the msmtools/flux docs and the
    references therein [1].

    References
    ----------
    [1] http://www.emma-project.org/v2.2.7/api/generated/msmtools.flux.tpt.html

    """
    def __init__(self, EvoMSM, source, target):
        self.msm = EvoMSM
        self.source = source
        self.target = target

        self.ReactiveFlux = tpt(self.msm.transition_matrix, self.source, self.target)
        self.net_flux = self.ReactiveFlux.net_flux
        self.total_flux = self.ReactiveFlux.total_flux
        self.forward_committor = self.ReactiveFlux.forward_committor
        self.backward_committor = self.ReactiveFlux.backward_committor

    def coarse_grain(self, sets):
        """Coarse grain flux based on a list of sets, which can represent metastable clusters

        Parameters
        ----------
        sets : list.
            List of set

        Returns
        -------
        coarse: Coarse grained ReactiveFlux object.

        """
        sets, coarse = self.ReactiveFlux.coarse_grain(sets)
        coarse.msm = self.msm
        coarse.sets = sets
        return coarse

    @property
    def source(self):
        """Get source node"""
        return self._source

    @source.setter
    def source(self, source):
        """Set source node/genotype to list of nodes(type=int) or genotypes(type=str)"""
        if isinstance(source, list):
            if not isinstance(source[0], int):
                df = self.msm.gpm.data
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
                df = self.msm.gpm.data
                self._target = [df[df['genotypes'] == t].index.tolist()[0] for t in target]
            elif isinstance(target[0], int):
                self._target = target
        else:
            raise Exception("Target has to be a list of at least one genotype(type=str) or node(type=int)")
