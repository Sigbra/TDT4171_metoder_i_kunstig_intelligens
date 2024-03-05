import numpy as np
from pathlib import Path
from typing import Tuple


class Node:
    """ Node class used to build the decision tree"""
    def __init__(self, attributes):
        self.children = {}
        self.parent = None
        self.attribute = attributes
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)

    def add_branch(self, attribute_value,node):
        self.children[attribute_value] = node


def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value

def entropy(examples):
    _, counts = np.unique(examples, return_counts=True)
    probabilities = counts /counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(examples, attribute, attribute_values):
    original_entropy_list = []
    for example in examples:
        original_entropy_list.append(example[-1])
    original_entropy_array = np.array(original_entropy_list)
    original_entropy = entropy(original_entropy_array)

    weighted_entropy_sum = 0
    for value in attribute_values:
        subset_value_list = []
        for example in examples:
            if example[attribute] == value:
                subset_value_list.append(example)
        subset = np.array(subset_value_list)

        subset_entropy_list = []
        for example in subset:
            subset_entropy_list.append(example[-1])
        subset_entropy_array = np.array(subset_entropy_list)
        subset_entropy = entropy(subset_entropy_array)

        weighted_entropy_sum += (len(subset / len(examples)) * subset_entropy)

    gain = original_entropy - weighted_entropy_sum
    return gain


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    if measure == 'random':
        return int(np.random.choice(attributes))
    elif measure == 'information_gain':
        gains = []
        for attribute in attributes:
            attribute_values = np.unique(examples[:, attribute])
            gain = information_gain(examples, attribute, attribute_values)
            gains.append(gain)

        return int(attributes[np.argmax(gains)])
    else:
        raise ValueError("Ops:) 'measure' has bad value")

def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # TODO implement the steps of the pseudocode in Figure 19.5 on page 678
    if len(examples)==0:
        node = Node(attributes=None)
        node.value = plurality_value(parent_examples)
        return node
    elif np.unique(examples[:, -1]).size == 1:
        node = Node(attributes=None)
        node.value = examples[0][-1]
        return node
    elif len(attributes)==0:
        node = Node(attributes=None)
        node.value = plurality_value(examples)
        return node
    else:
        A = importance(attributes, examples, measure)

        node = Node(attributes=A)
        if parent is not None:
            parent.children[branch_value] = node
        node.parent = parent

        for value in np.unique(examples[:, A]): #for each value v of A do:
            examples_i = np.array([e for e in examples if e[A] == value])
            attributes_i = [a for a in attributes if a!=A]
            subtree = learn_decision_tree(examples_i,
                                        attributes_i,
                                        examples,
                                        node,
                                        value,
                                        measure)
            node.add_branch(value, subtree)
        
    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test




if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "information_gain"
    #measure = "random"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")