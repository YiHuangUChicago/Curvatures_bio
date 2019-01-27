#!/usr/bin/env python3
"""
Calculate Wasserstein distance and barycenter for Gaussian Distributions
"""

import numpy as np
import numpy.linalg as LA


import sys
sys.path.append('../Util')
import Utils


def Wasserstein_dist_Gaussian(mu1, Sigma1, 
                              mu2, Sigma2):
    
    
    """
    
    """
    
    w_dist_sq = LA.norm(mu1 - mu2)**2
    sigma1 = Utils.mat_sqrt(Sigma1)
    sigma = Sigma1 + Sigma2 - 2. * Utils.mat_sqrt(np.matmul(np.matmul(sigma1, Sigma2), sigma1))
    w_dist_sq += np.trace(sigma)
    if w_dist_sq > 0:
        return np.sqrt(w_dist_sq)
    else:
        return 0


def _sigma_iteration(Sigmas, *, 
                     weights=None, 
                     num_rounds=1000, 
                     eps=1e-4, 
                     log=False):
    """
    
    """
    
    length = len(Sigmas)
    dim = Sigmas[0].shape[0]
    
    if weights is None:
        weights = np.ones(length)
    
    weights /= weights.sum()
    
    Sigma = np.identity(dim)
    log_dict = {}
    for r in range(num_rounds):
        sigma = Utils.mat_sqrt(Sigma)
        sigma_inv = LA.inv(sigma)
        summation = np.zeros((dim, dim))
        for i in range(length):
            s = Sigmas[i]
            summation += weights[i] * Utils.mat_sqrt(np.matmul(np.matmul(sigma, s), sigma))
        summation = np.matmul(summation, summation)
        Sigma_nxt = np.matmul(np.matmul(sigma_inv, summation), sigma_inv)
        diff = LA.norm(Sigma_nxt - Sigma)
        
        if log:
            log_dict[r] = diff

        if diff < eps:
            break
        else:
            Sigma = Sigma_nxt

    if log:
        return Sigma_nxt, log_dict
    else:
        return Sigma_nxt


def Wasserstein_barycenter_Gaussian(mus, Sigmas, *, 
                                    weights=None, 
                                    num_rounds=1000, 
                                    eps=1e-4):
    length = len(mus)
    dim = len(mus[0])

    if weights is None:
        weights = np.ones(length)
    weights_normalized = weights / np.sum(weights)
    
    mu_c = np.zeros(dim)
    for i in range(length):
        mu_c += weights_normalized[i] * mus[i]
    Sigma_c = _sigma_iteration(Sigmas, weights=weights_normalized, num_rounds=num_rounds, eps=eps)
    return mu_c, Sigma_c


if __name__ == '__main__':
    print('This is main of Wasserstein_Gaussian.py\n')