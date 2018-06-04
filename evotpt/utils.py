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


def count():
    counts = []
    for path in unique_sorted_paths():
        counts.append(self.paths.count(path))
    return tuple(counts)


def sort_paths_by_id(self):
    """Sort path  tuples by first to last position. """
    # Example: [(0, 2, 13, 24), (0, 1, 28, 30), (0, 2, 15, 22)] -> [(0, 1, 28, 30), (0, 2, 13, 24), (0, 2, 15, 22)].
    # Lamdba is an anonymous function that tells sorted() to take tup as input and then apply sorted on tup[:].
    # i.e. sort by all elements in tup. key=lambda tup: tup[1] would only sort on the second element in the tuple.
    sorted_pathlist = sorted(self.paths, key=lambda tup: tup[:])

    return sorted_pathlist

def unique_sorted_paths(self):
    """ Return unique sorted list of path IDs """
    uniq = list(set(self.paths))
    uniq_sorted_pathlist = sorted(uniq, key=lambda tup: tup[:])
    return uniq_sorted_pathlist