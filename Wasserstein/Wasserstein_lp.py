#!/usr/bin/env python3
import numpy as np 
import cvxpy as cvx
from scipy.optimize import linprog

"""
Wasserstein distance calculated using linear programming with
    - scipy.optimize.linprog
    - CVXPY package for convex optimization
"""

def Wasserstein_dist_lp(dist_matrix, mu, nu):
    """
    mu and nu are two discrete probability vectors
    Calculate Wasserstein dist by solving the linear programming problem
    """

    dim = len(mu)

    A_eq_mu = []
    for i in range(dim):
        A = np.zeros((dim, dim))
        A[i] = np.ones(dim)
        A_eq_mu.append(A)

    A_eq_mu = np.hstack(A_eq_mu)
    A_eq_nu = np.hstack([np.identity(dim)] * dim)
    A_eq = np.vstack([A_eq_mu, A_eq_nu])

    b_eq = np.vstack([mu.reshape(len(mu), 1), nu.reshape(len(nu), 1)])

    # print(A_eq)
    # print(b_eq)

    res = linprog(dist_matrix.flatten(),\
        A_ub=-np.identity(dim * dim), b_ub=np.zeros(dim * dim),\
        A_eq=A_eq, b_eq=b_eq)

    return res


def Wasserstein_dist_lp_dual(dist_matrix, mu, nu):
    """
    mu and nu are two discrete probability vectors
    Calculate Wasserstein dist by solving the dual linear programming problem
    """

    dim = len(mu)
    A_ub = []
    b_ub = []

    for i, j in combinations(range(dim), 2):
        vec = np.zeros(dim)
        vec[i] = 1.
        vec[j] = -1.
        A_ub.append(vec)
        A_ub.append(-vec)
        b_ub.append(dist_matrix[i, j])
        b_ub.append(-dist_matrix[i, j])

    c = mu - nu
    res = linprog(c, A_ub, b_ub)

    return res


def Wasserstein_dist_cvx(dist_matrix, mu, nu, p=1):
    """
    mu and nu are two discrete probability vectors
    Calculate Wasserstein dist by solving the convex optimization problem
    """
    
    num_rows = len(mu)
    num_cols = len(nu)
    dist = dist_matrix**p
    D = cvx.Variable((num_rows, num_cols))
    constraints = [\
        D >= 0, D <= 1,\
        # cvx.sum(D, axis=0) == np.reshape(nu, (1, num_cols)),\
        # cvx.sum(D, axis=1) == np.reshape(mu, (num_rows, 1)),\
        cvx.sum(D, axis=0) == np.array(nu),\
        cvx.sum(D, axis=1) == np.array(mu)]
    objective = cvx.Minimize(cvx.sum(cvx.multiply(dist, D)))
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return problem.value**(1. / p)


def Wasserstein_dist_cvx_dual(dist_matrix, mu, nu):
    """
    mu and nu are two discrete probability vectors
    Calculate Wasserstein dist by solving the convex optimization problem
    """

    f = cvx.Variable(num_vertices)

    constraints = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            constraints += [f[i] - f[j] <= dist_matrix[i][j], f[i] - f[j] >= -dist_matrix[i][j]]

    constraints += [cvx.sum(f) == 0]
    objective = cvx.Maximize(cvx.sum(cvx.multiply(mu - nu, f)))
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return problem.value


def Wasserstein_dist_matrix(dist_matrix, conditional_probs, p=1):
    """
    Calculate the Wasserstein distance matrix
    """
    
    num_vertices = len(dist_matrix)
    w_matrix = np.zeros((num_vertices, num_vertices))

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            print('({}, {})'.format(i, j))
            d = Wasserstein_dist(dist_matrix, conditional_probs[i], conditional_probs[j], p)
            w_matrix[i][j] = d
            w_matrix[j][i] = d

    return w_matrix


def Wasserstein_edge_weights(num_vertices, edges, pw_dist_matrix=None, edge_weights=None, verbose=False):
    """
    Calculate the Wasserstein edge weights with random walk probabilities and shortest path distance
    """
    if pw_dist_matrix is None:
        connected, pw_dist_matrix = pairwise_distances(num_vertices, edges, edge_weights)
    conditional_probs = hypergraph_random_walk(num_vertices, edges, edge_weights)
    
    w_edge_weights = []
    
    for i, edge in enumerate(edges):
        A = conditional_probs[edge, :].T
        barycenter = ot.bregman.barycenter(A, pw_dist_matrix, 0.1)

        sum_distances = 0
        for v in e:
            sum_distances += ot.sinkhorn2(conditional_probs[v], barycenter, pw_dist_matrix, 0.01)[0]

        w_edge_weights.append(sum_distances)

        if verbose:
            print('edge {}'.format(i))
            # print('\tbarycenter = {}'.format(barycenter))
            print('\tsum of distances = {}'.format(sum_distances))

    return np.array(w_edge_weights)


if __name__ == '__main__':
    print('This is main of Wasserstein_lp.py')