from msmtools.flux import pathways
from gpmap.utils import hamming_distance

def monotonic_incr(sequence, values):
    """See if a sequence of values is monotonically increasing.

    Parameters
    ----------
    sequence : list, tuple, 1D-array (dtype = int).
        Each element is an index for accessing a value from values.

    values : list, tuple, 1D-array (dtype = float, int).
        Stores all possible values that can occur in a sequence.

    Returns
    -------
    bool: True, False.
        True if sequence is monotonically increasing. Allows for neutral steps. False if not monotonically increasing.
    """

    for i in range(len(sequence)-1):
        if values[sequence[i]] > values[sequence[i+1]]:
            return False
    return True


class Pathways(object):
    def __init__(self, msm, flux_matrix, source, target, fraction=1, maxiter=1000):
        self.msm = msm
        self.source = source
        self.target = target
        self._pathways, self._capacities = pathways(flux_matrix, self.source, self.target, fraction=fraction, maxiter=maxiter)

        self._adaptive_paths = None

    def adaptive_paths(self):
        if self._adaptive_paths:
            return self._adaptive_paths
        else:
            adaptive_paths = []
            for path in self.pathways:
                if monotonic_incr(path, self.msm.gpm.data.fitnesses):
                    adaptive_paths.append(path)

            self._adaptive_paths = adaptive_paths

            return self._adaptive_paths

    def forward_paths(self, paths):
        fp = []
        min_dist = hamming_distance(self.msm.gpm.data.binary[self.source[0]], self.msm.gpm.data.binary[self.target[0]])

        for path in paths:
            if len(path) - 1 == min_dist:
                fp.append(path)

        return fp

    @property
    def pathways(self):
        return self._pathways

    @property
    def capacities(self):
        return self._capacities

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