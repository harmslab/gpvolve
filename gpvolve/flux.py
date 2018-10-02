from msmtools.flux import tpt as TPT


class tpt(object):
    def __init__(self, EvoMSM, source, target):
        self.msm = EvoMSM
        self.source = source
        self.target = target

        self.ReactiveFlux = TPT(self.msm.transition_matrix, self.source, self.target)
        self.net_flux = self.ReactiveFlux.net_flux
        self.total_flux = self.ReactiveFlux.total_flux
        self.forward_committor = self.ReactiveFlux.forward_committor
        self.backward_committor = self.ReactiveFlux.backward_committor

    def coarse_grain(self, sets):
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
