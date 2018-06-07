#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# -------------------------------------------------------------------------
# USAGE: mcmc_path_sampling.py <dataset>.json <outputfile>.json
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import sys
from scipy.stats import rv_discrete

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt.sampling import Sampling
from gpmap import GenotypePhenotypeMap


"""MARKOV CHAIN MONTE CARLO SIMULATION"""

class MarkovChainMonteCarlo(Sampling):
    """Class for sampling paths from genotype-phenotype map using the monte carlo method"""
    def __init__(self, gpm, outputfile, population_size, reversibility=True, null_steps=False, **kwargs):
        Sampling.__init__(self, gpm)

        # Set filename.
        self.outfilename = outputfile.split(".")[0]

        # Set population size.
        self.pop_size = population_size

        # Set reversibility
        self.reversibility = reversibility

        # Set null_step, i.e. is staying in the current state allowed or not.
        self.null_steps = null_steps

        # Get transition matrix
        self.tm = self.transition_matrix()

        print("Reverse steps allowed: %s. Default: True" % reversibility)

    def simulate(self):
        tm = self.tm

        # Get column names from transition matrix, i.e. all genotypes in transition matrix order)
        genotypes = list(tm)

        # Get integer index for each genotype starting at 0 (matches row indices).
        # There's probably a more pandas-like way to get row indices, but it works
        # as long as the matrix is square.
        gt_indices = [index for index in range(0, len(genotypes))]

        # Starting state is wildtype.
        StateCurr = self.binary_wildtype

        # Set up the log dictionary for storing paths and counting attempts.
        log = {}
        log["path"] = []
        log["path"].append(StateCurr)

        # Stop taking steps when the mutant genotype is reached.
        while StateCurr not in self.mutant:

            # Get probability mass function for current state/row, i.e. row vector of transition probabilities (sum=1).
            pmf = [probability for probability in tm.ix[genotypes.index(StateCurr), :]]
            # Draw a genotype based on the probability mass function.
            state_drawn = genotypes[rv_discrete(name='sample', values=(gt_indices, pmf)).rvs()]
            # Set drawn state as current state
            StateCurr = state_drawn

            # Append state to current path.
            log["path"].append(StateCurr)

        # Return the whole path.
        return log

gpm = GenotypePhenotypeMap.read_json(sys.argv[1])
markov_chain = MarkovChainMonteCarlo(gpm, outputfile=sys.argv[2],
                                     population_size=10000,
                                     reversibility=False, null_step=False)

markov_chain.sample(sys.argv[3], 1234)