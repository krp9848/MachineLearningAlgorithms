import numpy as np 
from collections import Counter

def entropy(labels:list) -> float:
    """Calculate the entropy of the given system

    Args:
        labels (list): Given state of the system

    Returns:
        float: Return the calculated entropy value of the given system 
    """
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError('Labels must be a list or numpy array')


    label_size = len(labels)
    labels_prob = [value/label_size for value in Counter(labels).values()]


    return -np.sum(labels_prob * np.log2(labels_prob))


def information_gain(original_labels: list, original_labels_split: list) -> float:
    """Calculates the information gain after the system is split into different parts

    Args:
        original_labels (list): Original dataset
        original_labels_split (list): List of splits of original dataset

    Returns:
        float: Information gain obtained after the split
    """
    return entropy(original_labels) - np.sum([(len(split) / len(original_labels)) * \
        entropy(split) for split in original_labels_split])



def build_tree():
    pass
