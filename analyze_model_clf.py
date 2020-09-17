import numpy as np
import torch
from torch.utils.data import DataLoader

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from src.datasets import Dataset, TorchDataset
from src.metrics import LT_dendrogram_purity
from src.LT_models import LTBinaryClassifier, LTClassifier
from src.optimization import evaluate

# LOAD_DIR = "./results/tab-datasets/HIGGS/depth=5/reg=0/seed=1337/"
# LOAD_DIR = "./results/clustering/GLASS/depth=4/reg=0/seed=1337/"
LOAD_DIR = "results/tabular/CLICK/depth=3/reg=596.9305372187174/mlp-layers=2/dropout=0.0043705947219156065/seed=1225/"

additional_load = {'checkpoint': None}
model = LTBinaryClassifier.load_model(LOAD_DIR, additional_load)
checkpoint = additional_load['checkpoint']

DATA_NAME = checkpoint['dataset']
SEED = checkpoint['seed']

MAX_DEPTH = 3
leaves = range(2**MAX_DEPTH - 1, 2**(MAX_DEPTH+1) - 1)

data = Dataset(DATA_NAME, seed=459107, normalize=True)

X, Y = data.X_test, data.y_test
NB_CLASSES = max(Y) + 1

score, class_hist = LT_dendrogram_purity(X, Y, model, model.latent_tree.bst, NB_CLASSES)

for c in range(NB_CLASSES):
    
    class_hist[c] = model.latent_tree.bst.normalize(class_hist[c])

# build graph for visualization
G = nx.from_numpy_array(model.latent_tree.bst.to_adj_matrix())
pos = graphviz_layout(G, prog='dot')

# plot a tree per class
for c in range(NB_CLASSES):

    plt.title(f'class {c}')

    nx.draw(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(class_hist[c])}, arrows=True, node_color=class_hist[c], cmap=plt.cm.PuBu)

    plt.savefig(f'{LOAD_DIR}class{c}.png')
    plt.clf()
