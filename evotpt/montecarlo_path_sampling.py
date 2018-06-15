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

from evotpt.sampling import Sampling
from evotpt import utils
from gpmap import GenotypePhenotypeMap

"""MONTE CARLO SIMULATION"""

class MonteCarlo(Sampling):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm, outputfile, population_size, reversibility=True, **kwargs):
        Sampling.__init__(self, gpm)

        # Set filename.
        self.outfilename = outputfile.split(".")[0]

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
            PhenoCurr, PhenoProp = utils.get_phenotype(self.data, StateCurr), utils.get_phenotype(self.data, StateProp)

            # Get random number from a uniform distribution between 0 and 1.
            rand_unif = random.uniform(0, 1)

            if utils.fixation_probability(self.data, PhenoCurr, PhenoProp, self.pop_size) > rand_unif:
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

    def propose_step(self, state_curr, reversibility):
        """ Propose the next step based on the current step's neighbors and whether reverse steps are allowed. """

        # Get list of neighbors of current genotype.
        neighbors = utils.get_neighbors(self.data, self.wildtype, state_curr, self.mutations, self.reversibility)

        # If reversibility is True, return the proposed state.
        if reversibility == True:
            pass

        # If reversibility is False, remove neighbors with hamming distance below 0, i.e. reverse steps.
        elif reversibility == False:
            for neighbor, index in zip(neighbors, range(0, len(neighbors)-1)):
                if utils.signed_hamming_distance(self.wildtype, state_curr, neighbor) < 0:
                    neighbors.pop(index)

        # Propose a neighbor randomly
        state_prop = random.choice(neighbors)

        return state_prop



gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
monte_carlo = MonteCarlo(gpm, outputfile=sys.argv[2], population_size=10000, reversibility=False)

monte_carlo.sample(sys.argv[3], 1234)