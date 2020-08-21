import numpy as np
import torch
from torch.utils.data import DataLoader

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from src.datasets import toy_dataset
from src.metrics import LT_dendrogram_purity
from src.tabular_datasets import Dataset
from src.LT_models import LTBinaryClassifier, LTClassifier
from src.optimization import evaluate
from src.utils import TorchDataset

# LOAD_DIR = "./results/tab-datasets/HIGGS/depth=5/reg=0/seed=1337/"
LOAD_DIR = "./results/clustering/GLASS/depth=4/reg=0/seed=1337/"

if 'xor' in LOAD_DIR:

    model = LTBinaryClassifier.load_model(LOAD_DIR)

    SEED = 1225
    np.random.seed(SEED)
    NB_CLASSES = 2

    # generate toy dataset
    X, Y = toy_dataset(100, 'xor')

else:
    DATA_NAME = LOAD_DIR.split('/')[-5]
    SEED = int(LOAD_DIR.split('/')[-2][5:])

    data = Dataset(DATA_NAME, random_state=SEED, normalize=True)

    X, Y = data.X_test, data.y_test
    NB_CLASSES = max(Y) + 1

    if NB_CLASSES == 2:
        model = LTBinaryClassifier.load_model(LOAD_DIR)
    else:
        model = LTClassifier.load_model(LOAD_DIR)

    testloader = DataLoader(TorchDataset(X, Y), batch_size=1024, shuffle=False)

    test_loss = evaluate(testloader, model, lambda x, y: (x != y).sum())
    print(f"test error rate: {test_loss}\n")

score, class_hist = LT_dendrogram_purity(X, Y, model, NB_CLASSES)

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
