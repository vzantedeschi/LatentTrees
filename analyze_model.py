import numpy as np
import torch
from torch.utils.data import DataLoader

import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

from src.datasets import toy_dataset
from src.tabular_datasets import Dataset
from src.LT_models import LTBinaryClassifier
from src.optimization import evaluate
from src.utils import TorchDataset

LOAD_DIR = "./results/tab-datasets/HIGGS/depth=5/reg=0/seed=1337/"
NB_CLASSES = 2
model = LTBinaryClassifier.load_model(LOAD_DIR)

if 'xor' in LOAD_DIR:
    SEED = 1225
    np.random.seed(SEED)

    # generate toy dataset
    X, Y = toy_dataset(100, 'xor')

else:
    DATA_NAME = LOAD_DIR.split('/')[-5]
    SEED = int(LOAD_DIR.split('/')[-2][5:])

    data = Dataset(DATA_NAME, random_state=SEED, quantile_transform=True, quantile_noise=1e-3, normalize=True)
    X, Y = data.X_test, data.y_test
    testloader = DataLoader(TorchDataset(X, Y), batch_size=BATCH_SIZE*2, shuffle=False)

    test_loss = evaluate(testloader, model, lambda x, y: (x != y).sum())
    print(f"test error rate: {test_loss}\n")

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