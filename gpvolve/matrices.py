import numpy as np
from .utils import hamming_distance

def hamming_distance_matrix(genotypes):
    """Create a pairwise Hamming distance matrix for a list of genotypes.
    """
    n = len(genotypes)
    hd_matrix = np.zeros((n,n))
    for i, g1 in enumerate(genotypes):
        for j, g2 in enumerate(genotypes):
            hd_matrix[i,j] = hamming_distance(g1, g2)
    return hd_matrix

def neighbor_matrix(genotypes):
    """Create a pairwise neighbor matrix for a set of genotypes. 1 for single-step
    neighbors and 0 for others.
    """
    # Build a hamming distance matrix
    hd_matrix = hamming_distance_matrix(genotypes)
    # Find single step neighbors
    neighbor_matrix = np.zeros(hd_matrix.shape)
    neighbor_matrix[hd_matrix==1] = 1
    return neighbor_matrix
