import numpy as np
import torch

import itertools
from scipy.stats import mode

def class_purity(dataset, true_y, model, bst, nb_classes):

    if isinstance(dataset, np.ndarray):
        
        zs, labels = model.predict_bst(torch.from_numpy(dataset).float())
    
    else:

        zs, labels = [], []
        for batch in dataset:
            # get tree representation of test points
            t_x, _ = batch
            z, l = model.predict_bst(t_x)

            zs.append(z)
            labels.append(l)

        zs, labels = np.vstack(zs), np.hstack(labels)        

    # get class histograms over tree nodes
    class_hist = np.empty((nb_classes, zs.shape[1]))

    for c in range(nb_classes):
        
        class_hist[c] = np.sum(zs[true_y == c] > 0, axis=0)

    purity = np.nan_to_num(class_hist / np.sum(class_hist, axis=0)) # node's fraction of points of a class

    return purity, class_hist, labels

def LT_dendrogram_purity(dataset, true_y, model, bst, nb_classes):

    purity, class_hist, labels = class_purity(dataset, true_y, model, bst, nb_classes)

    return dendrogram_purity(bst, labels, true_y, purity, nb_classes), class_hist

def dendrogram_purity(bst, pred_y, true_y, purity, nb_classes):

    # leaf assignment of points
    leaves = pred_y + bst.nb_split 
    
    num_pairs = 0
    score = 0
    
    c_point_leaves = {c: leaves[true_y == c] for c in range(nb_classes)}
    c_pairs = np.stack([np.array([sum(c_point_leaves[c] == n) for n in bst.leaves]) for c in range(nb_classes)])

    # loop over all possible combinations of outcomes
    for n1, n2 in itertools.product(bst.leaves, bst.leaves):

        a = bst.find_LCA(n1, n2)

        num_pairs += sum(c_pairs[:, n1 - bst.nb_split] * c_pairs[:, n2 - bst.nb_split])
        score += sum(purity[:, a] * c_pairs[:, n1 - bst.nb_split] * c_pairs[:, n2 - bst.nb_split])

    del c_point_leaves, c_pairs, purity

    return score / num_pairs

def node_statistics(X, Y, model, depth):

    x = torch.from_numpy(X).float()
    
    # get tree representation of test points
    zs, _ = model.predict_bst(x)
    tree_shape = zs.shape

    nb_nodes = 2**(depth+1) - 1
    selected_zs = zs[:, :nb_nodes] # compute only down to given depth

    # std of points assigned to each node
    stds = []
    for z_t in selected_zs.T:
        stds.append(np.std(X[z_t > 0]))

    # mean, std and mode of target values assigned to each node
    y_distrs = []
    for z_t in selected_zs.T:
        y_distrs.append(Y[z_t > 0])

    return np.stack(stds), y_distrs, zs


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

    purity = np.array([[1, 0] * bst1.nb_nodes])
    assert dendrogram_purity(bst1, Y+1, Y, purity, 2) == 1.0 

    purity = np.array([[0.5, 0.5] * bst1.nb_nodes])
    assert dendrogram_purity(bst1, Y+1, Y, purity, 2) == 0.5

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

    import time

    NB_CLASSES = 10
    NB_POINTS = 1000

    Y = np.array([0] * NB_POINTS)

    bst3 = BinarySearchTree(10)
    purity = np.zeros((NB_CLASSES, bst3.nb_nodes))

    t1 = time.time()
    dendrogram_purity(bst3, Y+1, Y, purity, NB_CLASSES)
    t2 = time.time()

    print(f"python: {t2-t1}s")
