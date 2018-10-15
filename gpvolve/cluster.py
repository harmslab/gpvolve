from msmtools.analysis import pcca
#from gpvolve.utils import cluster_peaks

class PCCA(object):
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

    def __init__(self, evomsm, m, *args, **kwargs):
        self.memberships = pcca(evomsm.transition_matrix, m)


class ClusterFromPaths(object):
    def __init__(self):
        pass


