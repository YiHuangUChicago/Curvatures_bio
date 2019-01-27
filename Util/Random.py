#!/usr/bin/env python3
import numpy as np
import numpy.linalg as LA


def rand_prob_vec(num_samples, dim, lower=0., upper=1.):
    """
    Generated random probability vector
    """
    prob_vec = np.random.uniform(lower, upper, size=(num_samples, dim))
    for pv in prob_vec:
        pv /= pv.sum()
    
    return prob_vec


def rand_orthogonal(dim):
    """
    Generate random orthogonal matrix
    """
    matrix = []
    for i in range(dim):
        vec = np.random.randn(dim)
        for j in range(len(matrix)):
            inner_prod = np.dot(vec, matrix[j])
            vec -= inner_prod * matrix[j]
        vec /= LA.norm(vec)
        matrix.append(vec)
    
    return np.array(matrix)


def rand_Gaussian(dim, radius=1., eigenval_low=0., eigenval_upper=1.):
    """
    Generate random Gaussian distribution
    """
    
    mu = np.random.uniform(-radius, radius, size=dim)
    
    eigenvals = np.random.uniform(eigenval_low, eigenval_upper, size=dim)
    o_matrix = rand_orthogonal(dim)
    Sigma = np.matmul(o_matrix.T, np.matmul(np.diag(eigenvals), o_matrix))
    
    return mu, Sigma


if __name__ == '__main__':
    print('This is main of Random.py\n')
    
    prob = rand_prob_vec(10, 3)
    print("10 probability vector of dimension 3 = \n{}\n".format(prob))
    
    o_matrix = rand_orthogonal(5)
    print("A random orthogonal matrix:\n{}\n".format(o_matrix))

    mu, Sigma = rand_Gaussian(5, 2., 1., 2.)
    print("A random Gaussian distribution of dimension 5, \
        \n\t with infinite norm of mu <= 2.0, \
        \n\t and with Sigma with eigenvalues between 1.0 and 2.0.")
    print("mu = \n{}".format(mu))
    print("Sigma = \n{}\n".format(Sigma))
