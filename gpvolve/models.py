import numpy as np
from .matrices import neighbor_matrix

def wright_fisher(genotypes, fitnesses, mu=1e-6, N=1e6, diploid=False):
    """Build a transition matrix for a wright_fisher model.
    """
    # Check that mutation rate and population size make sense.
    if mu * N > 1 :
        raise Exception("the population size, N, must be smaller than mutations rate, mu.")
    # Haploid/diploid matrix
    a = 1
    if diploid:
        a = 2
    # Get neighbor matrix
    nb_mat = neighbor_matrix(genotypes)
    # Build fitness matrices
    fitness_matrix_0 = np.multiply(fitnesses.reshape((-1,1)), nb_mat)
    fitness_matrix_1 = np.multiply(nb_mat, fitnesses)
    # Calculate a selection coefficient matrix
    selection_matrix = fitness_matrix_0 / fitness_matrix_1
    # Calculate fixation probabilities
    pi = np.nan_to_num((1-selection_matrix**a)/(1-(selection_matrix**(2*N))))
    W = 2 * a * mu * N * pi
    # Set the diagonal of the matrix to self probabilities
    W[np.diag_indices_from(W)] = 1 - W.sum(axis=1)
    return W
