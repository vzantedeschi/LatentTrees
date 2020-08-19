import numpy as np
import torch

import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

from src.datasets import toy_dataset
from src.LT_models import LTBinaryClassifier

LOAD_DIR = "./results/xor/depth=2/reg=0/"
NB_CLASSES = 2
SEED = 1225
np.random.seed(SEED)

model = LTBinaryClassifier.load_model(LOAD_DIR)

# generate toy dataset
X, Y = toy_dataset(100, 'xor')

# get tree representation of test points
zs, _ = model.predict_bst(torch.from_numpy(X).float())

# get class histograms over tree nodes
hist = np.empty((NB_CLASSES, zs.shape[1]))
for c in range(NB_CLASSES):
    
    hist[c] = np.sum(zs[Y == c] > 0, axis=0)

# build graph for visualization
G = nx.from_numpy_array(model.latent_tree.bst.to_adj_matrix())
pos = graphviz_layout(G, prog='dot')

# plot a tree per class
for c in range(NB_CLASSES):

    plt.title(f'class {c}')

    nx.draw(G, pos, labels={i: int(d) for i, d in enumerate(hist[c])}, arrows=True, node_color=hist[c], cmap=plt.cm.PuBu)

    plt.savefig(f'{LOAD_DIR}class{c}.png')
    plt.clf()