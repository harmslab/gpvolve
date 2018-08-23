from gpgraph import GenotypePhenotypeGraph


class MarkovModel(object):
    def __init__(self,
                 gpm,
                 model=None,
                 selection=1,
                 population_size=None,
                 two_step=False):

        # Set parameters
        self.gpm = gpm
        self.model = model
        self.population_size = population_size
        self.two_step = two_step

        # Apply selection to phenotypes in gpm
        self.apply_selection(gpm, selection)

    #     # Initialize GenotypePhenotypeGraph object and calculate transition matrix
    #     self.network = GenotypePhenotypeGraph(self.gpm)
    #     self._tm = self.tm
    #
    # def tm(self):
    #     if self._tm:
    #         return self._tm
    #     else:
    #         if self.two_step == True:
    #             pass  # Set edges between genotypes with hamming distances
    #
    #         self.network.add_model(model=self.model, population_size=self.population_size)
    #
    #         return T

    def apply_selection(self, gpm, selection):
        """Simualte varying selection pressure. The higher selection, the lower the selection pressure."""
        fitness = [ph + selection for ph in gpm.data.phenotypes]
        max_fit = max(fitness)
        norm_fitness = [fit / max_fit for fit in fitness]  # Normalize to 1
        gpm.data['fitnesses'] = norm_fitness
