import numpy as np
import time

from pqdict import minpq

def random_simple_graph(num_vertices, *, prob, directed=False, random_weight=False):
    
    """
    
    """
    
    adjacency_list = {i: {} for i in range(num_vertices)}
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                p = np.random.random()
                if p < prob:
                    if random_weight:
                        weight = np.random.random()
                    else:
                        weight = 1.
                    adjacency_list[i][j] = weight
                    if not directed:
                        adjacency_list[j][i] = weight
    
    return adjacency_list


# Kosaraju’s algorithm for finding strongly connected components
# Now the algorithm doesn't work for disconnected graph

def DFS(graph, root, stack, visited):
    visited.add(root)
    for neighbor in graph[root]:
        if neighbor not in visited:
            DFS(graph, neighbor, stack, visited)
    stack.append(root)


def reverse(graph):
    reversed_graph = {head: {} for head in graph.keys()}
    for head in graph.keys():
        for tail in graph[head]:
            if tail in reversed_graph:
                reversed_graph[tail][head] = graph[head][tail]
            else:
                reversed_graph[tail] = {head: graph[head][tail]}
    return reversed_graph


def stronglyConnectedComp(graph):
    
    """
    Kosaraju’s algorithm for finding the strongly connected components
    """
    
    stack = []
    visited = set()
    
    for vertex in graph.keys():
        if vertex not in visited: 
            DFS(graph, vertex, stack, visited)
    
    r_graph = reverse(graph)
    
    SCC = []
    visited = set()

    while len(stack) > 0:
        root = stack.pop()
        if root not in visited:
            scc = []
            DFS(r_graph, root, scc, visited)
            SCC.append(scc)
    
    return SCC


# Dijkstra algorithm for finding shortest path distance using priority dictionary

def dijkstra(graph, source, target=None, *, weight_dist_map=lambda x: x):
    
    """
    
    """
    
    dist = {}
    
    pq = minpq()
    for vertex in graph.keys():
        if vertex == source:
            pq[vertex] = 0
        else:
            pq[vertex] = float('inf')
            
    for vertex, min_dist in pq.popitems():
        if min_dist == float('inf'):
            break
            
        dist[vertex] = min_dist

        if vertex == target:
            break

        for neighbor in graph[vertex]:
            if neighbor in pq:
                
                new_dist = min_dist + weight_dist_map(graph[vertex][neighbor])
                if new_dist < pq[neighbor]:
                    pq[neighbor] = new_dist
            
    return dist


def Dijkstra(graph, source, target=None, *, weight_dist_map=lambda x: x):
    
    """
    
    """
    
    dist = {}
    parent = {}
    
    pq = minpq()
    for vertex in graph.keys():
        if vertex == source:
            pq[vertex] = 0
        else:
            pq[vertex] = float('inf')
            
    for vertex, min_dist in pq.popitems():
        if min_dist == float('inf'):
            break
            
        dist[vertex] = min_dist

        if vertex == target:
            break

        for neighbor in graph[vertex]:
            if neighbor in pq:
                
                new_dist = min_dist + weight_dist_map(graph[vertex][neighbor])
                if new_dist < pq[neighbor]:
                    pq[neighbor] = new_dist
                    parent[neighbor] = vertex
    
    return dist, parent

def induced_subgraph(graph, scc):
    subgraph = {head: {} for head in scc}
    scc_set = set(scc)
    
    for head in scc:
        for tail in graph[head]:
            if tail in scc_set:
                subgraph[head][tail] = graph[head][tail]
    
    return subgraph