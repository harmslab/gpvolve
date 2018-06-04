#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# -------------------------------------------------------------------------
# USAGE: montecarlo_path_sampling.py <dataset>.json <outputfile>.json <iteration number>
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import sys
import random

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evoTPT.sampling import Sampling
from gpmap import GenotypePhenotypeMap

"""MONTE CARLO SIMULATION"""

class MonteCarlo(Sampling):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm, outputfile, population_size, reversibility=True, **kwargs):
        Sampling.__init__(self, gpm)

        # Set filename.
        self._filename = outputfile

        # Set population size.
        self.pop_size = population_size

        # Set reversibility
        self.reversibility = reversibility

        print("Reverse steps allowed: %s. Default: True" % reversibility)

    def simulate(self):
        """Monte Carlo simulation"""

        stepcount = 0
        StateCurr = self.binary_wildtype

        log = {}
        log["path"] = []

        while StateCurr not in self.mutant:

            # Proposes random new state from neighbors of the current state.
            StateProp = self.propose_step(StateCurr, self.reversibility)

            # Get phenotypes of current and proposed genotype.
            PhenoCurr, PhenoProp = self.get_phenotype(StateCurr), self.get_phenotype(StateProp)

            # Get random number from a uniform distribution between 0 and 1.
            rand_unif = random.uniform(0, 1)

            if self.fixation_probability(PhenoCurr, PhenoProp, self.pop_size) > rand_unif:
                # Log current genotype
                log["path"].append(StateCurr)

                # Proposed state is accepted.
                StateCurr = StateProp
                stepcount += 1

            else:
                # Propose a different step instead, i.e. start over.
                continue
        # Log the last genotype.
        log["path"].append(StateCurr)
        return log

gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
monte_carlo = MonteCarlo(gpm, outputfile=sys.argv[2], population_size=10000, reversibility=False)

monte_carlo.sample(sys.argv[3], 1234)