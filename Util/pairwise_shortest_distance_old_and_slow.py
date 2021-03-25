def pairwise_distances(adjacency_list, *, edge_distances=None):
    
    """
    """
    
    vertices = list(adjacency_list.keys())
    num_vertices = len(vertices)
    
    nbh_distances = {}
    for vertice_from in adjacency_list:
        nbh_distances[vertice_from] = {vertice_from: 0.}
        for vertice_to in adjacency_list[vertice_from]:
            if edge_distances is None:
                edge_distance = 1.
            else:
                edge_distance = edge_distances[vertice_from][vertice_to]
            nbh_distances[vertice_from][vertice_to]=edge_distance

    is_updated = True
    while is_updated:
        is_updated = False
        for i in range(num_vertices):
            for j in range(num_vertices):
                vertice_from = vertices[i]
                vertice_to = vertices[j]
            
                path_distances = []
                for to, dist in nbh_distances[vertice_from].items():
                    if vertice_to in adjacency_list[to]:
                        if edge_distances is None:
                            edge_distance = 1.
                        else:
                            edge_distance = edge_distances[to][vertice_to]
                        path_distances.append(dist + edge_distance)
            
                if len(path_distances) > 0:
                    min_dist = min(path_distances)
                    if (vertice_to not in nbh_distances[vertice_from]) or \
                        (nbh_distances[vertice_from][vertice_to] > min_dist):
                        nbh_distances[vertice_from][vertice_to] = min_dist
                        is_updated = True
    
    return nbh_distances