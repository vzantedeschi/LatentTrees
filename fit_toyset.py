import hydra

import numpy as np
import pandas as pd
import torch

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from src.baselines import OptTree
from src.datasets import Dataset
from src.LT_models import LTBinaryClassifier, LTRegressor
from src.metrics import LT_dendrogram_purity
from src.monitors import MonitorTree
from src.optimization import train_batch, twophased_train_batch
from src.utils import deterministic

@hydra.main(config_path='config/default-xor.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.DISTR}/{cfg.model.TYPE}/twophased={cfg.training.TWO_PHASED}/split={cfg.model.SPLIT}/comp={cfg.model.COMP}/depth={cfg.model.BST_DEPTH}/seed={cfg.training.SEED}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve())

    deterministic(cfg.training.SEED)

    NORM = float('inf')

    # generate toy dataset
    data = Dataset(cfg.dataset.DISTR, n=cfg.dataset.N)

    if cfg.model.TYPE == 'LT':

        pruning = cfg.training.REG > 0

        if cfg.dataset.DISTR == 'reg-xor':
            
            model = LTRegressor(cfg.model.BST_DEPTH, 2, 1, pruned=pruning, linear=cfg.model.LINEAR, split_func=cfg.model.SPLIT, comp_func=cfg.model.COMP)

            # init loss
            criterion = torch.nn.MSELoss(reduction="mean")

        else:

            data.labels = data.Y

            # model = LTBinaryClassifier.load_model(SAVE_DIR) # to load a pretrained model instead
            model = LTBinaryClassifier(cfg.model.BST_DEPTH, 2, pruned=pruning, linear=cfg.model.LINEAR, split_func=cfg.model.SPLIT, comp_func=cfg.model.COMP)

            # init loss
            criterion = torch.nn.BCELoss(reduction="mean")

        # init optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.LR)

        monitor = MonitorTree(pruning, SAVE_DIR)

        if cfg.training.TWO_PHASED:
            twophased_train_batch(data.X, data.Y, model, optimizer, criterion, nb_iter=cfg.training.ITER, reg=cfg.training.REG, norm=NORM, monitor=monitor)
        else:
            train_batch(data.X, data.Y, model, optimizer, criterion, nb_iter=cfg.training.ITER, reg=cfg.training.REG, norm=NORM, monitor=monitor)

        monitor.close()
        # save model
        model.save_model(optimizer, dict(cfg), SAVE_DIR)
        bst = model.latent_tree.bst

    # baseline: Optimal Classification Trees https://www.mit.edu/~dbertsim/papers/Machine%20Learning%20under%20a%20Modern%20Optimization%20Lens/Optimal_classification_trees_MachineLearning.pdf
    elif cfg.model.TYPE == 'OPTREE':

        model = OptTree(bst_depth=cfg.model.BST_DEPTH, dim=2, N_min=1)
        model.train(data.X, data.Y)
        bst = model.bst

    else:
        raise NotImplementedError()

    # ---------------------------------------------------------------------- PLOT PREDICTIONS

    # create a mesh to plot in (points spread uniformly over the space)
    H = .02  # step size in the mesh
    x1_min, x1_max = data.X[:,0].min() - 0.1, data.X[:,0].max() + 0.1
    x2_min, x2_max = data.X[:,1].min() - 0.1, data.X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, H), np.arange(x2_min, x2_max, H)) # test points

    # estimate learned class boundaries
    test_x = np.c_[xx.ravel(), yy.ravel()]

    if cfg.model.TYPE == 'LT':

        t_x = torch.from_numpy(test_x).float()
        _, y_pred = model.predict_bst(t_x)

    else:
        y_pred = model.predict(test_x)
    
    y_pred = y_pred.reshape(xx.shape)

    score, class_hist = LT_dendrogram_purity(data.X, data.labels, model, bst, 2)
    print("Dendrogram purity:", score)

    # plot leaf boundaries
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.tab20c, alpha=0.6)

    # plot training points with true labels
    plt.scatter(data.X[data.labels == 0][:,0], data.X[data.labels == 0][:,1], s=20, marker="o", c='k')
    plt.scatter(data.X[data.labels == 1][:,0], data.X[data.labels == 1][:,1], s=20, marker="^", c='k')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title("{} dataset. lr={}; tree depth={}; iters={}".format(cfg.dataset.DISTR, cfg.training.LR, cfg.model.BST_DEPTH, cfg.training.ITER))

    plt.savefig(SAVE_DIR / f"{cfg.dataset.DISTR}.pdf", bbox_inches='tight', transparent=True)
    plt.clf()

    # ---------------------------------------------------------------------- PLOT TREES
    for c in range(2):
        
        class_hist[c] = bst.normalize(class_hist[c])

    # build graph for visualization
    G = nx.from_numpy_array(bst.to_adj_matrix())
    pos = graphviz_layout(G, prog='dot')

    # plot a tree per class
    for c in range(2):

        plt.title(f'class {c}')

        nx.draw(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(class_hist[c])}, arrows=True, node_color=class_hist[c], cmap=plt.cm.PuBu)

        plt.savefig(SAVE_DIR / f'class{c}.png', bbox_inches='tight', transparent=True)
        plt.clf()

if __name__ == "__main__":
    main()