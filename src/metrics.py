import numpy as np
import torch

import itertools
from scipy.stats import mode

def class_purity(dataset, true_y, model, bst, nb_classes):

    if isinstance(dataset, np.ndarray):
        
        zs, labels = model.predict_bst(torch.from_numpy(dataset))
    
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
        
    # loop over all possible combinations of outcomes
    for n1, n2 in itertools.product(bst.leaves, bst.leaves):

        a = bst.find_LCA(n1, n2)

        # loop over classes
        for c in range(nb_classes):

            c_point_leaves = leaves[true_y == c]
            c_pairs = sum(c_point_leaves == n1) * sum(c_point_leaves == n2)
            
            num_pairs += c_pairs
            score += purity[c, a] * c_pairs

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

    # distance from decision boundaries of points assigned to each node
    split_dists = model.db_distance(x).detach().numpy()

    all_dists = np.zeros(tree_shape)
    all_dists[:, model.latent_tree.bst.desc_left] = split_dists
    all_dists[:, model.latent_tree.bst.desc_right] = split_dists

    dist_medians, dist_means = [], []
    for z_t, dist_t in zip(selected_zs.T, all_dists[:, :nb_nodes].T):
        dist_medians.append(np.median(dist_t[z_t > 0]))
        dist_means.append(np.mean(dist_t[z_t > 0]))

    return np.stack(dist_medians), np.stack(dist_means), np.stack(stds), y_distrs, zs

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
