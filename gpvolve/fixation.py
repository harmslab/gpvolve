import numpy as np
import math


def strong_selection_weak_mutation(fitness1, fitness2):
    """Strong selection, weak mutation model."""
    sij = (fitness2 - fitness1) / fitness1
    if sij < 0:
        sij = 0
    return 1 - np.exp(-sij)


def ratio(fitness1, fitness2):
    sij = fitness1/fitness2
    return sij


def moran(fitness1, fitness2, population_size):
    """From Sella and Hirsh, 2005"""
    if fitness1 == fitness2:
        fitness2 = fitness2 - fitness1/1000

    sij = np.nan_to_num((1 - (fitness1 / fitness2)) / (1 - (fitness1 / fitness2) ** population_size))
    return sij


def mccandish(fitness1, fitness2, population_size):
    """From McCandish, 2011"""
    numer = 1 - math.e**(-2 * (fitness2 - fitness1))
    denom = 1 - math.e**(-2 * population_size * (fitness2 - fitness1))
    sij = numer/denom
    return sij

