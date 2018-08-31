

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
    b = 1/selection_gradient - 1
    fitness = [ph_ + b for ph_ in phenotypes]
    max_fit = max(fitness)
    # Normalize to 1
    norm_fitness = [fit / max_fit for fit in fitness]
    return norm_fitness

def one_to_one(phenotypes):
    """One-to-one mapping of phenotypes to fitnesses."""
    return phenotypes