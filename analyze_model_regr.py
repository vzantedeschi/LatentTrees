import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import seaborn as sns

from src.datasets import toy_dataset
from src.metrics import node_statistics
from src.tabular_datasets import Dataset
from src.LT_models import LTRegressor
from src.optimization import evaluate
from src.utils import TorchDataset

# LOAD_DIR = "./results/tab-datasets/HIGGS/depth=5/reg=0/seed=1337/"
# LOAD_DIR = "./results/reg-xor/LT/split=elu/depth=2/seed=2020/"
LOAD_DIR = "./results/clustering-selfsup/GLASS/out-feats=[7, 8]/depth=6/reg=829.99828226139/seed=6021991/"
# LOAD_DIR = "./results/tabular/YEAR/depth=3/reg=561.7353202746074/mlp-layers=3/dropout=0.07600075080048799/seed=1225/"

additional_load = {'checkpoint': None}
model = LTRegressor.load_model(LOAD_DIR, additional_load)
checkpoint = additional_load['checkpoint']

if 'reg-xor' in LOAD_DIR:

    SEED = 1225
    np.random.seed(SEED)
    NB_CLASSES = 2

    # generate toy dataset
    X, Y, _ = toy_dataset(100, 'reg-xor')

else:
    DATA_NAME = checkpoint['dataset']
    SEED = checkpoint['seed']

    data = Dataset(DATA_NAME, random_state=SEED, normalize=True)

    if DATA_NAME in ['GLASS', 'COVTYPE', 'ALOI']:
        out_features = list(map(int, LOAD_DIR.split('/')[4][10:].strip('][').split(',')))
        in_features = list(set(range(9)) - set(out_features))

        X = np.vstack((data.X_train[:, in_features], data.X_valid[:, in_features], data.X_test[:, in_features]))
        Y = np.vstack((data.X_train[:, out_features], data.X_valid[:, out_features], data.X_test[:, out_features]))

    else:
        X, Y = data.X_valid, data.y_valid

medians, means, stds, y_distrs = node_statistics(X, Y, model)

# build graph for visualization
G = nx.from_numpy_array(model.latent_tree.bst.to_adj_matrix())
pos = graphviz_layout(G, prog='dot')

for name, stat in {'median': medians, 'mean': means}.items():

    plt.title(f'Distance from decision boundaries: {name}')

    nx.draw(G, pos, arrows=True, node_color=stat, cmap=plt.cm.PuOr, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(stat)})

    plt.savefig(f'{LOAD_DIR}stat-{name}.png', bbox_inches='tight', transparent=True)
    plt.clf()

plt.title(f'Standard Deviations')

nx.draw(G, pos, arrows=True, node_color=stds, cmap=plt.cm.Reds, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(stds)})

plt.savefig(f'{LOAD_DIR}stat-std.png', bbox_inches='tight', transparent=True)
plt.clf()

pal = sns.color_palette("husl", model.latent_tree.bst.nb_leaves)
# plot target distributions
for t in model.latent_tree.bst.nodes:

    plt.title(f'Node {t}: target value distribution')

    for l in model.latent_tree.bst.leaves:

        if model.latent_tree.bst.is_ancestor(t, l):

            if Y.ndim == 1:
                sns.kdeplot(y_distrs[l], color=pal[l-model.latent_tree.bst.nb_split])
            
            else:
                sns.scatterplot(y_distrs[l][:, 0], y_distrs[l][:, 1], color=pal[l-model.latent_tree.bst.nb_split])

    plt.savefig(f'{LOAD_DIR}y-distr-{t}.png', bbox_inches='tight', transparent=True)
    plt.clf()
    break