#!/usr/bin/env python3
"""
Define functions that calculate
    - (normalized) hypergraph Laplacian
    - hypergraph random walk probabilities
    - neighborhood distance 
    - pairwise distance (shortest path distance)
"""
from itertools import product, combinations
import numpy as np


def hypergraph_Laplacian(
    num_vertices, 
    edges, *,
    edge_weights=None, 
    normalized=True):
    
    """
    Calculate the (normalized) hypergraph Laplacian.
    Let
        -_H_: Incidency matrix
        -_W_: Diagonal edge weight matrix
        -_D_e_: Diagonal edge degree matrix
        -_D_v_: Diagonal vertex degree matrix
    We have
        - Normalized hypergraph Laplacian: I - D_v^{-.5}HWD_e^{-1}H^{T}D_v^{-.5}
        - Unnoralized hypergraph Laplacian: D_v - HWD_e^{-1}H^{T}
    """
    
    if edge_weights is None:
        edge_weights = [1.] * len(edges)

    vertex_degrees = np.zeros(num_vertices)
    Laplacian = np.zeros((num_vertices, num_vertices))

    for i, edge in enumerate(edges):
        edge_degree = len(edge)
        for u in edge:
            vertex_degrees[u] += edge_weights[i]
            for v in edge:
                Laplacian[u][v] += edge_weights[i] / edge_degree 

    Laplacian = np.diag(vertex_degrees) - Laplacian

    if normalized == True:
        for u, v in product(range(num_vertices), repeat=2):
            Laplacian[u][v] /= (vertex_degrees[u] * vertex_degrees[v])**.5

    return Laplacian


def hypergraph_random_walk(
    num_vertices, 
    edges, *, 
    edge_weights=None,
    lazyness=None):
    
    """
    Calculate the (lazy) random-walk conditional probability vectors.
    Warning: We should ensure that the hypergraph 
        - does not have singleton edge;
        - does not have isolated vertex.
        Otherwise, exceptions will be raised.
    In case of:
        - lazyness is None:
            Conditioned on each edge, a vertex has equal probability 
            to jump to another vertex (include itself) contained in the same edge.
        - lazyness is a value between 0 and 1:
            Conditioned on each edge, a vertex has probability lazyness to not to jumpy,
            and equal probablity to jump to another vertex (no include itself)
            contained in the same edge.
    In both cases, the edges are weighted by edge_weights.
    """
    
    if edge_weights is None:
        edge_weights = [1.] * len(edges)

    vertex_degrees = np.zeros(num_vertices)
    conditional_probs = np.zeros((num_vertices, num_vertices))
    
    if lazyness is None:
        for i, edge in enumerate(edges):
            prob = edge_weights[i] / len(edge)
            for u in edge:
                for v in edge:
                    conditional_probs[u][v] += prob
    
    elif (lazyness < 0) or (lazyness > 1):
            raise Exception('Bad lazyness value! Lazyness has to be between 0 and 1.')
    else:  
        for i, edge in enumerate(edges):
            if len(edge) <= 1:
                raise Exception('edge {} is a singleton!')

            prob = edge_weights[i] / (len(edge) - 1.)
            for u in edge:
                for v in edge:
                    if v != u:
                        conditional_probs[u][v] += prob * (1. - lazyness)
                    else:
                        conditional_probs[u][v] += edge_weights[i] * lazyness 
    
    for u in range(num_vertices):
        total = conditional_probs[u].sum()
        if total == 0:
            raise Exception('vertex {} is isolated!'.format(u))
        else:
            conditional_probs[u] /= total
    
    return conditional_probs


def _neighborhood_distances(num_vertices, 
                           edges, *,
                           edge_distances=None):
    
    """
    Utility function of pairwise_distances.
    Not supposed to be used on its own!
    
    Calculate the distance between each pair of vertices that are connected.
    """
    
    if edge_distances is None:
        edge_distances = [1.] * len(edges)

    connected_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
    nbh_distances = np.zeros((num_vertices, num_vertices))
    
    for i, edge in enumerate(edges):
        for u, v in combinations(edge, 2):
            nbh_distances[u][v] += 1. / edge_distances[i]
            nbh_distances[v][u] += 1. / edge_distances[i]
            connected_matrix[u][v] += 1
            connected_matrix[v][u] += 1
    
    # Find the longest distance between two vertices that are connected.
    distances = []
    disconnected_set = set()
    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if connected_matrix[u][v] != 0:
                distance = 1. / nbh_distances[u][v]
                distances.append(distance)
                nbh_distances[u][v] = distance
                nbh_distances[v][u] = distance
            else:
                disconnected_set.add((u, v))
    
    # Use MAX to represent infinity distance between two vertices that are not connected.
    MAX = max(distances) * num_vertices + 1

    for p in disconnected_set:
        u, v = p[0], p[1]
        nbh_distances[u][v] = MAX
        nbh_distances[v][u] = MAX

    return nbh_distances, MAX


def pairwise_distances(num_vertices, 
                       edges, *,
                       edge_distances=None):
    """
    Calculate the shortest path distance matrix.
    """
    
    pw_distances, MAX = _neighborhood_distances(num_vertices, edges, edge_distances=edge_distances)

    updated = True
    while updated:
        updated = False
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                d = min([pw_distances[i][k] + pw_distances[k][j] for k in range(num_vertices)])
                if d < pw_distances[i][j]:
                    updated = True
                    pw_distances[i][j] = d
                    pw_distances[j][i] = d

    connected = pw_distances.max() < MAX
    
    return connected, pw_distances


if __name__ == '__main__':
    print('This is main of Hypergraph.py')