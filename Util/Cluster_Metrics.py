import numpy as np
from itertools import permutations

def entropy(probability):
    entropy = 0
    for p in probability:
        if p != 0:
            entropy -= p * np.log(p)
    return entropy


def joint_distribution(labels_1, labels_2):
    classes_1 = list(set(labels_1))
    classes_2 = list(set(labels_2))
    
    class_index_map_1 = {}
    for i in range(len(classes_1)):
        class_index_map_1[classes_1[i]] = i
    
    class_index_map_2 = {}
    for j in range(len(classes_2)):
        class_index_map_2[classes_2[j]] = j    
    
    joint_distribution = np.zeros((len(classes_1), len(classes_2)))
    for i in range(len(labels_1)):
        index_1 = class_index_map_1[labels_1[i]]
        index_2 = class_index_map_2[labels_2[i]]
        joint_distribution[index_1][index_2] += 1.
    
    joint_distribution /= joint_distribution.sum()
    
    return joint_distribution


def variation_information(labels_1, labels_2):
    jd = joint_distribution(labels_1, labels_2)
    return 2. * entropy(jd.flatten()) - entropy(jd.sum(axis=1)) - entropy(jd.sum(axis=0))


def labels_to_sets(labels):
    label_element_dict = {}
    for i in range(len(labels)):
        if labels[i] in label_element_dict:
            label_element_dict[labels[i]].append(i)
        else:
            label_element_dict[labels[i]] = [i]
    
    return {i: set(label_element_dict[i]) for i in label_element_dict}


def cluster_similarity(labels_1, labels_2):
    if len(labels_1) != len(labels_2):
        print("Different length")
        return

    D1 = labels_to_sets(labels_1)
    D2 = labels_to_sets(labels_2)
    K1 = list(D1.keys())
    K2 = list(D2.keys())
    
    perms = list(permutations(K2))
    max_sim = 0
    for perm in perms:
        count = 0
        for i in range(len(K1)):
            count += len(D1[K1[i]] & D2[perm[i]])
        max_sim = max(count / float(len(labels_1)), max_sim)
    
    return 1. - max_sim

if __name__ == '__main__':
    print('This is main function of cluster_metrics.py')