import numpy as np

from functools import reduce

class BinarySearchTree():

    def __init__(self, depth=2):

        # Example: decision tree of depth=2
        # max num nodes T = 2^(depth+1) - 1 = 7 nodes.
        #
        #      0
        #    /   \
        #   1     2
        #  / \   / \
        # 3   4 5   6

        self.depth = depth

        self.nb_nodes = 2**(depth+1) - 1
        self.nodes = range(self.nb_nodes)

        self.nb_split = 2**depth - 1
        self.split_nodes = range(self.nb_split)

        self.leaves = range(self.nb_split, self.nb_nodes)
        self.nb_leaves = len(self.leaves)

        self.desc_left = range(1, self.nb_nodes, 2)
        self.desc_right = range(2, self.nb_nodes, 2)

    def __str__(self):
        return str_as_bst(self.nodes)

    def parent(self, n):
        
        if n == 0 or n >= self.nb_nodes:
            return None

        return (n - 1) // 2

    def predict(self, z):
        """ each leaf corresponds to a class """
        import pdb; pdb.set_trace()
        labels = np.argmax(z[:, self.leaves], 1)

        return labels

def str_as_bst(nodes):

    depth = int((len(nodes) + 1) ** 0.5)
    lines = []

    # compute last line first to fix line width
    lines = [str(reduce(lambda n1, n2: str(n1).zfill(2) + " " + str(n2).zfill(2), nodes[2**depth - 1: 2**(depth + 1) - 1]))]
    len_lines = len(lines[0])

    for h in range(depth):

        spacing = max(1, int(len_lines / (2**(depth - h - 1))) - 1)

        l = " " * (spacing // 2)
        l += str(reduce(lambda n1, n2: str(n1).zfill(2) + " "*spacing + str(n2).zfill(2), nodes[2**(depth - h - 1) - 1: 2**(depth - h) - 1]))
        l += " " * (spacing // 2)

        lines.append(l)

    return reduce(lambda l1, l2: l1 + '\n' + l2, lines[::-1])