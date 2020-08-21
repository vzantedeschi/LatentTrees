import numpy as np
import torch

import itertools

def LT_dendrogram_purity(X, Y, model, nb_classes):

    # get tree representation of test points
    zs, labels = model.predict_bst(torch.from_numpy(X).float())

    # get class histograms over tree nodes
    class_hist = np.empty((nb_classes, zs.shape[1]))

    for c in range(nb_classes):
        
        class_hist[c] = np.sum(zs[Y == c] > 0, axis=0)

    purity = np.nan_to_num(class_hist / np.sum(class_hist, axis=0)) # node's fraction of points of a class

    return dendrogram_purity(model.latent_tree.bst, labels, Y, purity, nb_classes), class_hist

def dendrogram_purity(bst, pred_y, true_y, purity, nb_classes):

    # leaf assignment of points
    leaves = pred_y + bst.nb_split 
    
    num_pairs = 0
    score = 0

    # loop over classes
    for c in range(nb_classes):

        # loop over all pairs of points of that class
        for n1, n2 in itertools.product(leaves[true_y == c], leaves[true_y == c]):

            num_pairs += 1
            a = bst.find_LCA(n1, n2)
            score += purity[c, a]

    return score / num_pairs

if __name__ == "__main__":

    from math import isclose

    from trees import BinarySearchTree

    NB_CLASSES = 1
    NB_POINTS = 10

    Y = np.array([0] * NB_POINTS)
    bst1 = BinarySearchTree(4)
    
    # highest score when all points are assigned to the same leaf
    purity = np.array([[1] * bst1.nb_nodes])
    assert dendrogram_purity(bst1, Y, Y, purity, NB_CLASSES) == 1.0
    assert dendrogram_purity(bst1, Y+1, Y, purity, NB_CLASSES) == 1.0  

    # lowest score when purity is null
    purity = np.array([[0] * bst1.nb_nodes])
    assert dendrogram_purity(bst1, Y, Y, purity, NB_CLASSES) == 0.

    purity = np.array([[0.1] * bst1.nb_nodes])
    assert isclose(dendrogram_purity(bst1, Y, Y, purity, NB_CLASSES), 0.1)

    NB_POINTS = 4
    bst2 = BinarySearchTree(2)
    Y = np.array([0] * NB_POINTS)

    num_pairs = NB_POINTS ** 2

    purity = np.array([[0, 1, 0] + [0] * (bst2.nb_nodes - 3)])
    pred_y = np.array([0, 1, 2, 3])

    assert isclose(dendrogram_purity(bst2, pred_y, Y, purity, NB_CLASSES), 2 / num_pairs), (dendrogram_purity(bst2, pred_y, Y, purity, NB_CLASSES), 2 / num_pairs)

    purity = np.array([[0.5, 1, 0] + [0] * (bst2.nb_nodes - 3)])
    assert isclose(dendrogram_purity(bst2, pred_y, Y, purity, NB_CLASSES), (2 * 1 + 0.5 * 8) / num_pairs), (dendrogram_purity(bst2, pred_y, Y, purity, NB_CLASSES), (2 * 1 + 0.5 * 8) / num_pairs)

    purity = np.array([[0.5, 1, 0, 1, 1]  + [0] * (bst2.nb_nodes - 5)])

    assert isclose(dendrogram_purity(bst2, pred_y, Y, purity, NB_CLASSES), (4 * 1 + 0.5 * 8) / num_pairs), (dendrogram_purity(bst2, pred_y, Y, purity, NB_CLASSES), (4 * 1 + 0.5 * 8) / num_pairs)
