import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from src.datasets import toy_dataset
from src.metrics import node_statistics
from src.tabular_datasets import Dataset
from src.LT_models import LTRegressor
from src.optimization import evaluate
from src.utils import TorchDataset

# LOAD_DIR = "./results/tab-datasets/HIGGS/depth=5/reg=0/seed=1337/"
LOAD_DIR = "./results/reg-xor/linear=False/depth=2/reg=0/seed=2020/"
# LOAD_DIR = "./results/clustering/GLASS/depth=4/reg=0/seed=1337/"

model = LTRegressor.load_model(LOAD_DIR)

if 'reg-xor' in LOAD_DIR:

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

    testloader = DataLoader(TorchDataset(X, Y), batch_size=1024, shuffle=False)

    criterion = MSELoss(reduction="sum")

    test_loss = evaluate(testloader, model, lambda x, y: (x != y).sum())
    print(f"test MSE: {test_loss}\n")

medians, means, stds = node_statistics(X, model)

# build graph for visualization
G = nx.from_numpy_array(model.latent_tree.bst.to_adj_matrix())
pos = graphviz_layout(G, prog='dot')

# plot a tree per class
for name, stat in {'median': medians, 'mean': means}.items():

    plt.title(f'Distance from decision boundaries: {name}')

    nx.draw(G, pos, arrows=True, node_color=stat, cmap=plt.cm.PuOr, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(stat)})

    plt.savefig(f'{LOAD_DIR}stat-{name}.png')
    plt.clf()

plt.title(f'Standard Deviations')

nx.draw(G, pos, arrows=True, node_color=stds, cmap=plt.cm.Reds, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(stds)})

plt.savefig(f'{LOAD_DIR}stat-std.png')
plt.clf()
