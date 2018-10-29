import numpy as np
from scipy.sparse import csr_matrix
from .utils import rm_self_prob

def to_greedy(transition_matrix):
    """Turn transition matrix into 'greedy' transition matrix. Only the step with the highest positive fitness
    difference is allowed (prob. = 1), all other steps are not permitted (prob. = 0)

    Parameters
    ----------
    transition_matrix : 2D numpy.ndarray.
        Transition matrix where the highest value T(i->j) per row i should correspond to the step s(i->j) where j is
        the neighbor of genotype i with the highest fitness. Can be obtained using the 'ratio' fixation function, where
        transition probability T(i->j) is simply the ratio of fitness j over fitness i.

    Returns
    -------
    M : 2D numpy.ndarray.
        Transition matrix corresponding to a 'greedy random walk' on the genotype-phenotype map.

    References
    ----------
    de Visser JA, Krug J. 2014. Empirical fitness landscapes and the predictability of evolution. Nature Reviews
    Genetics 15:480–490.
    """
    T = transition_matrix.copy()
    # Remove self-looping probability/matrix diagonal = 0
    np.fill_diagonal(T, 0)

    # Get column index of max value for each row.
    indices = np.argmax(T, axis=1)
    # Set index pointer (check scipy.sparse.csr_matrix documentation).
    indptr = np.array(range(T.shape[0] + 1))
    # Since there is only on possible greedy step per row, it is assigned probability of 1.
    data = np.ones(T.shape[0])

    M = csr_matrix((data, indices, indptr), shape=T.shape).toarray()

    return M


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
    sij : 1D numpy.ndarray(dtype=float).
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

    # If identical fitnesses found, add a 1.000.000ths of the second fitness value to itself to marginally unlevel the two.
    for index in err_idx:
        copy = fitness2[index]
        fitness2[index] = copy + copy / 10 ** 6

    # Calculate fixation probability.
    sij = np.nan_to_num((1 - (fitness1/fitness2)) / (1 - pow(fitness1/fitness2, population_size)))
    return sij


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
    sij : 1D numpy.ndarray(dtype=float).
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

    sij = (1 - np.exp(-2 * (fitness2-fitness1))) / (1 - np.exp(-2 * population_size * (fitness2-fitness1)))
    return sij


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
    sij = preference2 / preference1

    # Set fixation probability to one for neutral or beneficial mutations.
    sij[sij > 1] = 1

    # Apply beta factor.
    sij = sij ** beta

    return sij


def strong_selection_weak_mutation(fitness1, fitness2):
    """Strong selection, weak mutation model."""
    sij = (fitness2 - fitness1) / fitness1
    if sij < 0:
        sij = 0
    return 1 - np.exp(-sij)


def ratio(fitness1, fitness2):
    """Fixation probability equals the ratio of new fitness over old fitness"""
    sij = fitness2 / fitness1
    return sij


def equal_fixation(fitness1, fitness2):
    """Only adaptive steps are allowed and all adaptive steps have the same probability"""
    sij = fitness2 / fitness1
    sij[sij <= 1] = 0
    sij[sij > 1] = 1
    return sij


