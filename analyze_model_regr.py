import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import seaborn as sns

from src.datasets import toy_dataset
from src.metrics import node_statistics, class_purity
from src.datasets import Dataset, TorchDataset
from src.LT_models import LTRegressor
from src.optimization import evaluate

# LOAD_DIR = "./results/clustering-selfsup/GLASS/out-feats=[7, 8]/depth=6/reg=829.99828226139/seed=1225/"
# LOAD_DIR = "./results/tabular/MICROSOFT/depth=8/reg=784.2480977010307/mlp-layers=3/dropout=0.10054922066470592/seed=1225/"
LOAD_DIR = "./results/clustering-selfsup/GLASS/out-feats=[0, 1]/depth=6/reg=17.893973029582362/seed=1225/"

additional_load = {'checkpoint': None}
model = LTRegressor.load_model(LOAD_DIR, additional_load)
checkpoint = additional_load['checkpoint']

DATA_NAME = checkpoint['dataset']
SEED = checkpoint['seed']

MAX_DEPTH = 3
leaves = range(2**MAX_DEPTH - 1, 2**(MAX_DEPTH+1) - 1)

if DATA_NAME in ['GLASS', 'COVTYPE', 'ALOI']:
    out_features = list(map(int, LOAD_DIR.split('/')[4][10:].strip('][').split(',')))
    in_features = list(set(range(9)) - set(out_features))

    data = Dataset(DATA_NAME, in_features=in_features, out_features=out_features, seed=459107, normalize=True)

    X = np.vstack((data.X_train_in, data.X_valid_in, data.X_test_in))
    Y = np.vstack((data.X_train_out, data.X_valid_out, data.X_test_out))
    true_y = np.hstack((data.y_train, data.y_valid, data.y_test))

else:

    data = Dataset(DATA_NAME, seed=459107, normalize=True)

    X, Y = data.X_valid, data.y_valid
    true_y = Y

NB_CLASSES = np.max(true_y) + 1

medians, means, stds, y_distrs, zs = node_statistics(X, Y, model, MAX_DEPTH)

# build graph for visualization
G = nx.from_numpy_array(model.latent_tree.bst.to_adj_matrix(MAX_DEPTH))
pos = graphviz_layout(G, prog='dot')

for name, stat in {'median': medians, 'mean': means}.items():

    plt.title(f'Distance from decision boundaries: {name}')

    nx.draw(G, pos, arrows=True, node_color=stat, cmap=plt.cm.PuOr, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(stat)})

    plt.savefig(f'{LOAD_DIR}stat-{name}.png', bbox_inches='tight', transparent=False)
    plt.clf()

plt.title(f'Standard Deviations')

nx.draw(G, pos, arrows=True, node_color=stds, cmap=plt.cm.Reds, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(stds)})

plt.savefig(f'{LOAD_DIR}stat-std.png', bbox_inches='tight', transparent=False)
plt.clf()

pal = sns.color_palette("husl", len(leaves))

if DATA_NAME in ['GLASS', 'COVTYPE', 'ALOI', 'MICROSOFT']:
    _, class_hist, _ = class_purity(X, true_y, model, model.latent_tree.bst, NB_CLASSES)
    class_hist = class_hist[:, :2**(MAX_DEPTH+1) - 1]

    for c in range(NB_CLASSES):
        
        class_hist[c] = model.latent_tree.bst.normalize(class_hist[c], MAX_DEPTH)

    # plot a tree per class
    for c in range(NB_CLASSES):

        plt.title(f'class {c}')

        nx.draw(G, pos, arrows=True, node_color=class_hist[c], cmap=plt.cm.PuBu, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=6, labels={i: np.round(d, decimals=2) for i, d in enumerate(class_hist[c])})

        plt.savefig(f'{LOAD_DIR}class{c}.png', bbox_inches='tight')
        plt.clf()

# plot target distributions
for t in model.latent_tree.bst.nodes:

    plt.title(f'Node {t}: target value distribution')

    empty = True
    for l in leaves:

        if len(y_distrs[l]) > 1 and model.latent_tree.bst.is_ancestor(t, l):

            empty = False

            if Y.ndim == 1:

                try:
                    sns.kdeplot(y_distrs[l], color=pal[l - 2**MAX_DEPTH - 1], shade=True, shade_lowest=False)
                except:
                    sns.distplot(y_distrs[l], color=pal[l - 2**MAX_DEPTH - 1], kde=False)
            
            # elif y_distrs[l].size > 4:
            #     sns.kdeplot(y_distrs[l][:, 0], y_distrs[l][:, 1], color=pal[l - 2**MAX_DEPTH - 1], shade=True, shade_lowest=False)

            else:
                sns.scatterplot(y_distrs[l][:, 0], y_distrs[l][:, 1], color=pal[l - 2**MAX_DEPTH - 1])

    if not empty:

        if Y.ndim == 1:
            plt.xlim(np.min(Y), np.max(Y))

        else:

            plt.ylim(np.min(Y[:, 1]) - 2, np.max(Y[:, 1]) + 2)
            plt.xlim(np.min(Y[:, 0]) - 2, np.max(Y[:, 0]) + 2)

        plt.savefig(f'{LOAD_DIR}y-distr-{t}.png', bbox_inches='tight', transparent=False)
        plt.clf()

if DATA_NAME in ['GLASS', 'COVTYPE', 'ALOI', 'MICROSOFT']:
    import pandas as pd
    predictions = model.latent_tree.bst.get_nodes_level(zs, MAX_DEPTH)[:, None]

    df = pd.DataFrame(data=np.hstack((Y, X, predictions)), columns=list(range(9)) + ["class"])

    sns.set(style="ticks")

    sns.pairplot(df, hue='class', palette=pal, diag_kind='hist')

    plt.tight_layout()
    plt.savefig(f'{LOAD_DIR}all-feats.png', bbox_inches="tight")