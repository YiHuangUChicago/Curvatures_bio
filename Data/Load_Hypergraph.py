#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def _get_vertices(num_vertices, edges):
	vertex_dict = {i: [] for i in range(num_vertices)}
	for i, edge in enumerate(edges):
		for v in edge:
			vertex_dict[v].append(i)
	
	vertices = [sorted(vertex_dict[v]) for v in vertex_dict]

	return vertices


def load_UCI_Zoo():
	df = pd.read_csv("~/Desktop/Curvatures/Data/Datasets/UCI_Zoo.txt", sep=',')
	features = [\
		'hair', 'feathers', 'eggs', 'milk',\
		'airborne', 'aquatic', 'predator', 'toothed',\
		'backbone', 'breathes', 'venomous', 'fins',\
		'legs', 'tail', 'domestic', 'catsize']

	edge_names = {}
	edges = []
	edge_count = 0

	for f in features:
		values = np.sort(df[f].unique())
		for val in values:
			edge = list(df.index[df[f] == val].values)
			edges.append(edge)
			edge_name = f + '_' + str(val)
			edge_names[edge_count] = edge_name
			edge_count += 1

	vertex_labels = {}
	vertex_names = {}

	for idx in df.index:
		vertex_labels[idx] = df['type'][idx]
		vertex_names[idx] = df['name'][idx]

	num_vertices = len(vertex_labels)
	num_edges = len(edges)

	vertices = _get_vertices(num_vertices, edges)
	
	Graph = {'num_vertices': num_vertices,
			'num_edges': num_edges,
			'vertices': vertices,
			'edges': edges,
			'vertex_labels': vertex_labels,
			'vertex_names': vertex_names,
			'edge_names':edge_names}

	return Graph


def load_UCI_Congress():
	df = pd.read_csv("~/Desktop/Curvatures/Data/Datasets/UCI_Congressional_Voting_Records.txt", sep=',')
	features = [\
		'handicapped-infants', 'water-project-cost-sharing',\
		'adoption-of-the-budget-resolution', 'physician-fee-freeze',\
		'el-salvador-aid', 'religious-groups-in-schools',\
		'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',\
		'mx-missile', 'immigration',\
		'synfuels-corporation-cutback', 'education-spending',\
		'superfund-right-to-sue', 'crime',\
		'duty-free-exports', 'export-administration-act-south-africa']
	
	edge_names = {}
	edges = []
	edge_count = 0
	
	for f in features:
		for val in ['y', 'n']:
			edge = list(df.index[df[f] == val].values)
			edges.append(edge)
			edge_name = f + '_' + val
			edge_names[edge_count] = edge_name
			edge_count += 1
   
	vertex_labels = {}
	for idx in df.index:
		vertex_labels[idx] = df['class'][idx]

	num_vertices = len(vertex_labels)
	num_edges = len(edges)

	vertices = _get_vertices(num_vertices, edges)
	
	Graph = {'num_vertices': num_vertices,
			'num_edges': num_edges,
			'vertices': vertices,
			'edges': edges,
			'vertex_labels': vertex_labels,
			'edge_names':edge_names}

	return Graph


def load_UCI_Mushroom(num_edible=None, num_poisonous=None):
	df = pd.read_csv("~/Desktop/Curvatures/Data/Datasets/UCI_Mushroom.txt", sep=',')
	features = [\
		'cap-shape', 'cap-surface', 'cap-color',\
		'bruises', 'odor', 'gill-attachment',\
		'gill-spacing', 'gill-size', 'gill-color',\
		'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',\
		'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',\
		'veil-type', 'veil-color', 'ring-number',\
		'ring-type', 'spore-print-color', 'population', 'habitat']

	if (num_edible is not None) or (num_poisonous is not None):
		e_indices = []
		p_indices = []
		vertex_dict = {}
		for idx in df.index:
			edibility = df['edibility'][idx]
			vertex_dict[idx] = {'edibility': edibility}
			if edibility == 'e':
				e_indices.append(idx)
			else:
				p_indices.append(idx)

		if num_edible is not None:
			e_indices = np.random.choice(e_indices, num_edible, replace=False)
		if num_poisonous is not None:
			p_indices = np.random.choice(p_indices, num_poisonous, replace=False)

		chosen_indices = list(e_indices) + list(p_indices)
		df = df[df.index.isin(chosen_indices)]
		df.sort_values(by=['edibility'], inplace=True)
		df.index = range(len(df))
	
	edge_names = {}
	edges = []
	edge_count = 0
	
	for f in features:
		values = np.sort(df[f].unique())
		for val in values:
			edge = list(df.index[df[f] == val].values)
			edges.append(edge)
			edge_name = f + '_' + val
			edge_names[edge_count] = edge_name
			edge_count += 1
	
	vertex_labels = {}
	for idx in df.index:
		vertex_labels[idx] =  df['edibility'][idx]

	num_vertices = len(vertex_labels)
	num_edges = len(edges)

	vertices = _get_vertices(num_vertices, edges)
	
	Graph = {'num_vertices': num_vertices,
			'num_edges': num_edges,
			'vertices': vertices,
			'edges': edges,
			'vertex_labels': vertex_labels,
			'edge_names':edge_names}

	return Graph


if __name__ == '__main__':
	print('This is the main function of Load_Hypergraph.')
	# Graph = load_UCI_Zoo()
	# Graph = load_UCI_Congress()
	# Graph = load_UCI_Mushroom(200, 200)
	# print(Graph)
