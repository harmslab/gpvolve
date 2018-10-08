# from msmtools.analysis import pcca
# #from gpvolve.utils import cluster_peaks
#
# class PCCA(object):
#     """Runs PCCA++ [1] to compute a metastable decomposition of MSM states.
#
#     Parameters
#     ----------
#     c : int.
#         Desired number of metastable sets.
#
#     Returns
#     -------
#     Nothing : None.
#         The important metastable attributes are set automatically.
#
#     Notes
#     -----
#     The metastable decomposition is done using the pcca method of the pyemma.msm.MSM class.
#     For more details and references: https://github.com/markovmodel/PyEMMA/blob/devel/pyemma/msm/models/msm.py
#     """
#
#     def __init__(self, evomsm, m, *args, **kwargs):
#         super().__init__(evomsm.transition_matrix, m, *args, **kwargs)
#
#         self.msm = evomsm
#         #self.cluster_peaks = cluster_peaks(self.msm, self.P.metastable_sets)  # After clustering one can use the cluster peaks as source and sink for tpt.
#
#
# class ClusterFromPaths(object):
#     def __init__(self):
#         pass
#
#
