import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from src.baselines import OptTree
from src.datasets import toy_dataset
from src.metrics import LT_dendrogram_purity
from src.optimization import train_batch
from src.utils import make_directory

DISTR = "xor"
N = 100
TREE_DEPTH = 3
LR = 0.1
ITER = 1e4
REG = 0
NORM = float('inf')
SAVE_DIR = f"./results/optree/{DISTR}/depth={TREE_DEPTH}/"
make_directory(SAVE_DIR)

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# generate toy dataset
X, Y = toy_dataset(N, DISTR)

# # train latent class tree and logistic regressor
# model, optimizer = train_batch(X, Y, bst_depth=TREE_DEPTH, nb_iter=ITER, lr=LR, reg=REG, norm=NORM, root_dir=SAVE_DIR)

# # save model
# model.save_model(optimizer, {'bst_depth': TREE_DEPTH, 'in_size': 2, 'pruned': REG > 0, 'lr': LR, 'nb_iter': ITER, 'distr': DISTR, 'point': N, 'seed': SEED}, SAVE_DIR)

# baseline: Optimal Classification Trees https://www.mit.edu/~dbertsim/papers/Machine%20Learning%20under%20a%20Modern%20Optimization%20Lens/Optimal_classification_trees_MachineLearning.pdf

model = OptTree(bst_depth=TREE_DEPTH, dim=2, N_min=1)
model.train(X, Y)

# ---------------------------------------------------------------------- PLOT PREDICTIONS
# define colors (looks good also in printed grey scales)
colors = [(1, 1, 1), (0.5, 0.5, 1)]
cm = LinearSegmentedColormap.from_list('twocolor', colors, N=100)

# create a mesh to plot in (points spread uniformly over the space)
H = .02  # step size in the mesh
x1_min, x1_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
x2_min, x2_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, H), np.arange(x2_min, x2_max, H)) # test points

# estimate learned class boundaries
test_x = np.c_[xx.ravel(), yy.ravel()]

# t_x = torch.from_numpy(test_x).float()
# y_pred = model.predict(t_x).numpy()
# y_pred = y_pred.reshape(xx.shape)

y_pred = model.predict(test_x)
y_pred = y_pred.reshape(xx.shape)

score, class_hist = LT_dendrogram_purity(X, Y, model, model.bst, 2)
print(score)

# plot class boundaries
plt.contourf(xx, yy, y_pred, cmap=cm, alpha=0.6)

# plot training points with true labels
plt.scatter(X[Y == 0][:,0], X[Y == 0][:,1], cmap=cm, s=20, marker="o", edgecolors=colors[1], c=colors[0])
plt.scatter(X[Y == 1][:,0], X[Y == 1][:,1], cmap=cm, s=20, marker="^", c=colors[1])

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.title("{} dataset. lr={}; tree depth={}; iters={}".format(DISTR, LR, TREE_DEPTH, ITER))

plt.savefig(SAVE_DIR + DISTR + ".pdf", bbox_inches='tight', transparent=True)
plt.clf()

for c in range(2):
    
    class_hist[c] = model.bst.normalize(class_hist[c])

# build graph for visualization
G = nx.from_numpy_array(model.bst.to_adj_matrix())
pos = graphviz_layout(G, prog='dot')

# plot a tree per class
for c in range(2):

    plt.title(f'class {c}')

    nx.draw(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(class_hist[c])}, arrows=True, node_color=class_hist[c], cmap=plt.cm.PuBu)

    plt.savefig(f'{SAVE_DIR}class{c}.png', bbox_inches='tight', transparent=True)
    plt.clf()