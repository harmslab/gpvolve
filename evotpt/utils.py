#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import json
import sys
import random
from math import e
from scipy.stats import rv_discrete
from operator import mul
from functools import reduce
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from gpmap import GenotypePhenotypeMap
from gpmap.utils import hamming_distance


def signed_hamming_distance(self, current, proposed):
    """Return the signed Hamming distance between equal-length sequences """
    nonbinary = []

    # Get non-binary version of the genotypes
    for genotype in [current, proposed]:
        # First: get index of binary genotype in the dataframe.
        genotype_index = self.data.index[self.data['binary'] == genotype].tolist()
        # Second: Pull out the corresponding binary from the dataframe and set as binary neighbor.
        nonbinary.append(self.data.iloc[genotype_index[0]]['genotypes'])

    # Count differences between wt and each genotype
    current_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(nonbinary[0], self.wildtype))
    proposed_to_wt = sum(ch1 != ch2 for ch1, ch2 in zip(nonbinary[1], self.wildtype))
    # Get the signed hamming distance between the two genotypes.
    # e.g. +1 if proposed states has one mutation more than the current state.
    signed_hamming = proposed_to_wt - current_to_wt

    return signed_hamming