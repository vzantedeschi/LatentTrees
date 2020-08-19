import numpy as np
import torch

import itertools

def dendrogram_purity(X, Y, model, num_classes):

    # get tree representation of test points
    zs, labels = model.predict_bst(torch.from_numpy(X).float())

    # get class histograms over tree nodes
    class_hist = np.empty((num_classes, zs.shape[1]))

    for c in range(num_classes):
        
        class_hist[c] = np.sum(zs[Y == c] > 0, axis=0)

    # leave assignment of points
    leaves = labels + model.latent_tree.bst.nb_split

    purity = np.nan_to_num(class_hist / np.sum(class_hist, axis=0)) # node's fraction of points of a class
    
    num_pairs = 0
    score = 0

    # loop over classes
    for c in range(num_classes):

        # loop over all pairs of points of that class
        for n1, n2 in itertools.product(leaves[Y == c], leaves[Y == c]):
            
            num_pairs += 1
            a = model.latent_tree.bst.find_LCA(n1, n2)
            score += purity[c, a]

    return score / num_pairs, class_hist
