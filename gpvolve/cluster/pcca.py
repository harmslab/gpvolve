from .base import GenotypePhenotypeClusters
from .utils import *
import msmtools.analysis as mana
import warnings


class PCCA(GenotypePhenotypeClusters):
    """Runs PCCA++ [1] to compute a metastable decomposition of MSM states.

    Parameters
    ----------
    m : int.
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
    def __init__(self, gpmsm, m):
        self.gpmsm = gpmsm
        self.m = m
        # Compute membership vectors.
        memberships = mana.pcca_memberships(gpmsm.transition_matrix, self.m)
        assignments = cluster_assignments(memberships)
        clusters = cluster_sets(assignments)

        super().__init__(self.gpmsm, clusters)
        self.memberships = memberships
        self.assignments = assignments
