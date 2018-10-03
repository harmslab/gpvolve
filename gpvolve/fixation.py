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
    """Computes the fixation probability between two 1D arrays of fitnesses or two single fitnesses.

    Parameters
    ----------
    fitness1 : 1D numpy.ndarray(dtype=float).
        Fitnesses of the first node of all edges. Fitness of genotype that the population is homogeneous for.

    fitness2 : 1D numpy.ndarray(dtype=float).
        Fitnesses of the second node of all edges. Fitness of genotype that "tries" to get fixed in the population.

    population_size : int.
        Population genetics parameter that is inversely proportional to the level of genetic drift.

    Returns
    -------
    fix : 1D numpy.ndarray(dtype=float).
        Returns array with all fixation probabilities of the system in the order defined by fitness1 and fitness2.

    Notes
    -----
    The mathematical function is not robust to the case where two fitnesses are identical(division by 0 error).
    This python function checks for such cases and adds a 1.000.000ths of the second fitness value to itself in order
    to marginally unlevel the two fitnesses. Note that, although unlikely, this might effect evolutionary trajectories
    in an unpredictable way.

    References
    ----------
    G. Sella, A. E. Hirsh: The application of statistical physics to evolutionary biology, Proceedings of the National
    Academy of Sciences Jul 2005, 102 (27) 9541-9546.
    """
    # Check for pairs of identical fitnesses.
    err_idx = np.where(fitness1 - fitness2 == 0)
    # If identical fitnesses found, add a 1.000.000 ths of the second fitness value to itself to marginally unlevel the two.
    for index in err_idx:
        copy = fitness2[index]
        fitness2[index] = copy + copy / 10 ** 6

    # Calculate fixation probability.
    fix = np.nan_to_num((1 - (fitness1/fitness2)) / (1 - pow(fitness1/fitness2, population_size)))
    return fix


def mccandlish(fitness1, fitness2, population_size):
    """Computes the fixation probability between two 1D arrays of fitnesses or two single fitnesses.

    Parameters
    ----------
    fitness1 : 1D numpy.ndarray(dtype=float).
        Fitnesses of the first node of all edges. Fitness of genotype that the population is homogeneous for.

    fitness2 : 1D numpy.ndarray(dtype=float).
        Fitnesses of the second node of all edges. Fitness of genotype that "tries" to get fixed in the population.

    population_size : int.
        Population genetics parameter that is inversely proportional to the level of genetic drift.

    Returns
    -------
    fix : 1D numpy.ndarray(dtype=float).
        Returns array with all fixation probabilities of the system in the order defined by fitness1 and fitness2.

    References
    ----------
    McCandlish, D. M. (2011), VISUALIZING FITNESS LANDSCAPES. Evolution, 65: 1544-1558.
    """
    # Check for pairs of identical fitnesses.
    err_idx = np.where(fitness1 - fitness2 == 0)
    # If identical fitnesses found, add a 1.000.000 ths of the second fitness value to itself to marginally unlevel the two.
    for index in err_idx:
        copy = fitness2[index]
        fitness2[index] = copy + copy / 10 ** 6

    fix = (1 - np.exp(-2 * (fitness2-fitness1))) / (1 - np.exp(-2 * population_size * (fitness2-fitness1)))
    return fix


def bloom(preference1, preference2, beta=1):
    """Computes probability (F(r_x->r_y)) of fixing amino acid x at site r when site r is amino acid y using
    amino acid preference data from deep-mutational scanning experiments.

    Parameters
    ----------
    preferences1 : 1D numpy.ndarray(dtype=float, int).
        Array of amino acid preferences. The ith element corresponds to the preference of amino acid x at site r.

    preferences2 : 1D numpy.ndarray(dtype=float, int).
        Array of amino acid preferences. The ith element corresponds to the preference of amino acid y at site r.

    beta : int, float (beta >= 0).
        Free parameter that scales the stringency of amino acid preferences. Beta = 1: Equal stringency of deep
        mutational scanning experiments and natural evolution. Beta < 1: Less stringent than natural selection.
        Beta > 1: More stringent than natural selection.

    Returns
    -------
    fix : 1D numpy.ndarray(dtype=float).
        1D array of fixation probabilities.

    References
    ----------
    Equation 3 - Jesse D. Bloom, Molecular Biology and Evolution, Volume 31, Issue 10, 1 October 2014, Pages 2753–2769
    """
    # Calculate preference ratios.
    fix = preference2 / preference1

    # Set fixation probability to one for neutral or beneficial mutations.
    fix[fix > 1] = 1

    # Apply beta factor.
    fix = fix ** beta
    return fix



