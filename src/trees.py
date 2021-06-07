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

    def is_ancestor(self, n1, n2):

        return self.find_LCA(n1, n2) == n1

    def predict(self, z):
        """ each leaf corresponds to a class """
        labels = np.argmax(z[:, self.leaves], 1)

        return labels

    def to_adj_matrix(self, depth=None):

        adj_matrix = np.zeros((self.nb_nodes, self.nb_nodes))
        adj_matrix[self.split_nodes, self.desc_left] = 1
        adj_matrix[self.split_nodes, self.desc_right] = 1

        if depth is None:
            return adj_matrix
        else:
            return adj_matrix[:2**(depth+1) - 1, :2**(depth+1) - 1]

    def normalize(self, z, depth=None):
        """ normalize nodes at same depth """
        if depth is None:
            depth = self.depth

        for d in range(depth + 1):

            i = 2 ** d - 1
            z[i:i + 2 ** d] /= sum(z[i:i + 2 ** d])

        return np.nan_to_num(z)

    def find_LCA(self, n1, n2):
        """ find Lowest Common Ancestor between two nodes of the tree """

        while n1 != n2:
            n1, n2 = min(n1, n2), max(n1, n2)

            n2 = self.parent(n2)

        return n1

    def get_nodes_level(self, z, depth=0):

        leaves = np.argmax(z[:, self.leaves], 1) + self.nb_split

        res = leaves
        for i in range(self.depth - depth):
            res = (res - 1) // 2

        return res 

    def get_node_ancestors(self, n):

        ancs = [n]

        while n > 0:
            n = self.parent(n)
            ancs.append(n)

        return ancs

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

if __name__ == "__main__":

    bst = BinarySearchTree(2)

    assert bst.find_LCA(3, 4) == 1, bst.find_LCA(3, 4)
    assert bst.find_LCA(5, 6) == 2, bst.find_LCA(5, 6)
    assert bst.find_LCA(0, 6) == 0, bst.find_LCA(0, 6)
    assert bst.find_LCA(1, 6) == 0, bst.find_LCA(1, 6)

    assert bst.get_nodes_level(np.array([[0, 0, 0, 1, 0, 0, 0]]), 0) == 0, bst.get_nodes_level(np.array([[0, 0, 0, 1, 0, 0, 0]]), 0)
    assert bst.get_nodes_level(np.array([[0, 0, 0, 1, 0, 0, 0]]), 1) == 1, bst.get_nodes_level(np.array([[0, 0, 0, 1, 0, 0, 0]]), 1)
    assert bst.get_nodes_level(np.array([[0, 0, 0, 1, 0, 0, 0]]), 2) == 3, bst.get_nodes_level(np.array([[0, 0, 0, 1, 0, 0, 0]]), 2)
    assert bst.get_nodes_level(np.array([[3, 1, 1, 1, 0, 2, 0]]), 1) == 2, bst.get_nodes_level(np.array([[0, 0, 0, 1, 0, 0, 0]]), 1)

    bst = BinarySearchTree(4)

    assert bst.find_LCA(3, 4) == 1, bst.find_LCA(3, 4)
    assert bst.find_LCA(5, 6) == 2, bst.find_LCA(5, 6)
    assert bst.find_LCA(0, 6) == 0, bst.find_LCA(0, 6)
    assert bst.find_LCA(1, 6) == 0, bst.find_LCA(1, 6)
    assert bst.find_LCA(25, 27) == 2, bst.find_LCA(25, 27)