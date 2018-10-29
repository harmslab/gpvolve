import numpy as np

def linear_skew(phenotypes, selection_gradient):
    """Transforms the phenotype data to match the given selection gradient and stores them as gpm.data.phenotypes.

    Parameters
    ----------
    selection_gradient : float, int. (default=1)
        The selection gradient defines the slope of the phenotype-fitness function.

    Returns
    -------
    norm_fitness : list.
        List of normalized fitnesses
    """

    ### USE NUMPY SYNTAX TO SPEED UP. INPUT SHOULD BE NUMPY ARRAY ANYWAY ###
    b = 1/selection_gradient - 1
    fitness = [ph_ + b for ph_ in phenotypes]
    max_fit = max(fitness)
    # Normalize to 1
    norm_fitness = [fit / max_fit for fit in fitness]
    return norm_fitness


def exponential(phenotypes, exponent=1.):
    """Apply exponential fitness function to phenotypes

    Parameters
    ----------
    phenotypes : 1D numpy.ndarray.
        Phenotype values (dtype=float).

    exponent : float, int (default=1).
        Exponent by which the phenotypes will be raised.

    Returns
    -------
    fitnesses : 1D numpy.ndarray.
        List of fitnesses.
    """
    fitnesses = phenotypes.copy()
    # Raising 0 phenotypes to a power will result in NaN, hence we change any NaN to 0.
    fitnesses = np.nan_to_num(fitnesses**exponent)

    return fitnesses


def one_to_one(phenotypes):
    """One-to-one mapping of phenotypes to fitnesses."""
    fitnesses = phenotypes.copy()
    return fitnesses


def step_function(phenotypes, interval, function='floor'):
    """Define intervals and assign same fitness to all phenotypes within the same itnerval.
    Example: phenotypes = [0.38, 0.32, 0.41], interval = 0.1, function = 'floor' --> fitnesses = [0.3, 0.3, 0.4]

    Parameters
    ----------
    phenotypes : list/1D numpy.ndarray.
        Phenotype values (dtype=float).

    interval : float.
        Interval within which phenotypes get the same fitness.

    function : str, 'floor' or 'ceil' (default='floor').
        Function to define if lower value ('floor') or upper value ('ceil') of the specific interval is used as fitness.
        Example (interval = 0.1):
        [0.48, 0.42, 0.32] --'floor'--> [0.4, 0.4, 0.3]
        [0.48, 0.42, 0.32] --'ceil'--> [0.5, 0.5, 0.4]

    Returns
    -------
    fitnesses : list/1D numpy.ndarray.
        Fitness values (dtype=float).
    """
    pass
