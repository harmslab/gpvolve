#!/Users/leandergoldbach/miniconda3/bin/python

# Author: Leander Goldbach / Major parts copied from:
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/flux/api.py and
# https://github.com/markovmodel/msmtools/blob/devel/msmtools/analysis/sparse

# -------------------------------------------------------------------------
# OUTSIDE IMPORTS
# -------------------------------------------------------------------------

import numpy as np
import sys

# -------------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------------

from evotpt.visualization import GenotypePhenotypeGraph
from evotpt import utils
from gpmap import GenotypePhenotypeMap

def number_of_paths(pmf):
    nb_paths = len(pmf)
    return nb_paths

def adaptive_paths(gpm, pmf):
    ap_pmf = {}
    for path, prob in pmf.items():
        phens = [utils.get_phenotype(gpm.data, gt) for gt in path]
        # if the minimum phenotype is not at position 0, then the path is not adaptive
        minm = min(phens)
        minima = [i for i, x in enumerate(phens) if x == minm]
        if phens.index(min(phens)) == 0 and len(minima) == 1:
            ap_pmf[path] = prob
    return ap_pmf




