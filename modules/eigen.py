import numpy as np
from typing import Tuple
from numpy import linalg as LA

def second_smallest_eigenvalue(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the second smallest eigenvalue and its corresponding eigenvector of a given matrix A.

    Args:
        A (np.ndarray): The matrix A.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The second smallest eigenvalue and its corresponding eigenvector.
    """
    #We first obtain the eigenvalues and eigenvectors of the matrix
    w, v = LA.eig(A)
    #We sort the eigenvalues
    sorted_w = np.sort(w)
    #We obtain the index of the second smallest eigenvalue
    index = np.where(w == sorted_w[1])
    #We obtain the second smallest eigenvalue and its corresponding eigenvector
    second_smallest_eigenvalue = w[index]
    second_smallest_eigenvector = v[:,index]
    #We return the second smallest eigenvalue and its corresponding eigenvector
    return second_smallest_eigenvalue[0], second_smallest_eigenvector[:, 0]

def normalized_laplacian(A: np.ndarray, d: int) -> np.ndarray:
    """
    This function computes the normalized laplacian matrix of a given d-regular graph with adjacency matrix A.

    Args:
        A (np.ndarray): The adjacency matrix of the graph.
        d (int): The degree of the graph.

    Returns:
        np.ndarray: The normalized laplacian matrix.
    """
    #Here we define the identity matrix
    I = np.eye(A.shape[0])
    #Here we return the normalized laplacian matrix
    return I - (1/d)*A

def sort_indexes(eigenvector: np.ndarray) -> np.ndarray:
    """
    This function returns the indexes of the sorted entries of a given eigenvector.

    Args:
        eigenvector (np.ndarray): The eigenvector.

    Returns:
        np.ndarray: The indexes of the sorted entries of the eigenvector.
    """
    #We obtain the indexes of the sorted entries of the eigenvector
    indexes = np.argsort(eigenvector, axis=0)
    #Here we return just the list of indexes
    indexes = indexes[:,0]
    return indexes
