import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.datasets import toy_dataset
from src.optimization import train_batch
from src.utils import make_directory

DISTR = "xor"
N = 100
TREE_DEPTH = 4
LR = 0.1
ITER = 1e4
REG = 0.001
NORM = 1 # 1, 0 or float('inf')
SAVE_DIR = "./results/"

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# generate toy dataset
X, Y = toy_dataset(N, DISTR)

# train latent class tree and logistic regressor
model = train_batch(X, Y, bst_depth=TREE_DEPTH, nb_iter=ITER, lr=LR, reg=REG, norm=NORM)

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
t_x = torch.from_numpy(test_x).float()

# x = torch.cat((t_x, torch.ones((len(t_x), 1))), 1)
# print(str_as_bst(model.sparseMAP(x).detach().numpy()[0]))

y_pred = model.predict(t_x).numpy()
y_pred = y_pred.reshape(xx.shape)

# plot class boundaries
plt.contourf(xx, yy, y_pred, cmap=cm, alpha=0.6)

# plot training points with true labels
plt.scatter(X[Y == 0][:,0], X[Y == 0][:,1], cmap=cm, s=20, marker="o", edgecolors=colors[1], c=colors[0])
plt.scatter(X[Y == 1][:,0], X[Y == 1][:,1], cmap=cm, s=20, marker="^", c=colors[1])

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.title("{} dataset. lr={}; tree depth={}; iters={}".format(DISTR, LR, TREE_DEPTH, ITER))

make_directory(SAVE_DIR)
plt.savefig(SAVE_DIR + DISTR + ".pdf", bbox_inches='tight', transparent=True)

plt.show()