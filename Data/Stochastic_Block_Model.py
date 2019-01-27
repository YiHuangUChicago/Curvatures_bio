#!/usr/bin/env python3
"""
Define functions:
    - generate_random_communities
    - stochastic_block_model
"""
import numpy as np
from numpy.random import choice, uniform, shuffle
from itertools import combinations


def generate_random_communities(
	num_vertices,*, 
	num_communities=None, 
	proportions=None, 
	shuffled=False):
	"""
	Generate random communities with given proportions
	- if proportions is not None:
		- length of proportion must equal to num_communities;
		- proportions must have sum 1.
	- if proportions is None:
		- num_communities must not be None;
		- generate communities as evenly as possible.
	- value of shuffle
		- False: communities compose of consecutive numbers
		- True: communities compose of shuffled numbers.
	Return:
		communities = [community_1, ...., community_k]
		where each community is a list
	"""

    # Sanity Check of num_communities and proportions:
	if proportions is None:
		if num_communities is None:
			raise Exception('num_communities and proportions cannot both be None.')
		else:
			proportions = np.ones(num_communities) / float(num_communities)
	else:
		if sum(proportions) != 1.:
			proportions /= proportions.sum()
		elif num_communities is None:
			num_communities = len(proportions)
		elif num_communities != len(proportions):
			raise Exception('num_communities must has the same length as proportions')

	# Generate sizes of each community:
	sizes = []
	for p in proportions:
		size = int(num_vertices * p)
		sizes.append(size)
	offset = num_vertices - sum(sizes)
	if offset > 0:
		if offset > num_communities:
			raise Exception('generate random communities fail!')
		to_add_indices = choice(num_communities, offset)
		for idx in to_add_indices:
			sizes[idx] += 1

	# Generate communities:
	vertices = np.arange(num_vertices)
	if shuffled:
		shuffle(vertices)
	start = 0
	communities = []
	for i in range(num_communities):
		communities.append(list(vertices[start: start + sizes[i]]))
		start += sizes[i] 

	return communities


def stochastic_block_model(communities, uniformality, prob):
	"""
	Given communities, generate uniform hypergraph with given uniformality.
    Suppose prob = [.4, .2, .1], then 
        - a hyperedge with vertices come from one community will have probability .4 to exist
        - a hyperedge with vertices come from two communities will have probability .2 to exist
        - a hyperedge with vertices come from three communities will have probability .1 to exist
        - a hyperedge with vertices come from more communities than three cannot exist
	"""
	if len(communities) < len(prob):
		print("WARNING: few communities than length of the probability vector")
	else:
		prob = list(prob) + (len(communities) - len(prob)) * [0]
	community_map = {}
	num = 0
	for i in range(len(communities)):
		num += len(communities[i])
		for v in communities[i]:
			community_map[v] = i

	edges = []
	for vertex_set in combinations(range(num), uniformality):
		comm = [community_map[v] for v in vertex_set] 
		diversity = len(set(comm))
		key = uniform()
		if key < prob[diversity - 1]:
			edges.append(list(vertex_set))
			
	return edges


if __name__ == '__main__':
	print("This is the main function of Stochastic_Block_Model.py")

	# # Test generate_random_communities:
	# num_vertices = np.random.randint(100)
	# num_communities = np.random.randint(5, 10)
	# print('number of vertices = {}\nnumber of communities = {}'.format(num_vertices, num_communities))
	# 
	# proportions = np.random.uniform(0, 1, size=num_communities)
	# proportions /= proportions.sum()
	# print('proportions = {}'.format(proportions))

	# communities = generate_random_communities(num_vertices, num_communities=num_communities)
	# 
	# for i, community in enumerate(communities):
	# 	print('community {}: {}'.format(i, community))

	# # Test stochastic_block_model:
	# communities = generate_random_communities(50, num_communities=3)
	# for i, community in enumerate(communities):
	# 	print('community {}: {}'.format(i, community))
	# edges = stochastic_block_model(communities, 3, [.01, .005])
	# for e in edges:
	# 	print(e)
