import numpy as np
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt

from src.datasets import toy_dataset
from src.monitors import MonitorTree
from src.LT_models import LTLinearRegressor
from src.optimization import train_batch
from src.utils import make_directory

DISTR = "reg-xor"
N = 1000
TREE_DEPTH = 2
LR = 0.2
ITER = 1e3
ETA = 0

SAVE_DIR = "./results/{}/eta={}/".format(DISTR, ETA)

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

pruning = (ETA is not None)
nb_nodes = (2**(TREE_DEPTH + 1) - 1)
nb_leaves = 2**TREE_DEPTH

monitor = MonitorTree(pruning, SAVE_DIR)

# generate toy dataset
X, Y, labels = toy_dataset(N, DISTR)

# 3 input features, 1 target value
model = LTLinearRegressor(TREE_DEPTH, 3, 1, eta=[ETA]*nb_nodes)

# init optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# init loss
criterion = torch.nn.MSELoss(reduction="mean")

# cast to pytorch Tensors
t_y = torch.from_numpy(Y[:, None]).float()
t_x = torch.from_numpy(X).float()

# create a mesh to plot in (points spread uniformly over the space)
H = .02  # step size in the mesh
x1_min, x1_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
x2_min, x2_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, H), np.arange(x2_min, x2_max, H)) # test points

test_x = np.c_[xx.ravel(), yy.ravel()]
test_x = torch.from_numpy(test_x).float()

model.train()

pbar = tqdm(range(int(ITER)))
for i in pbar:

    # print(model.sparseMAP.eta.detach().numpy())
    optimizer.zero_grad()

    y_pred = model(t_x)

    loss = criterion(y_pred, t_y)
    pbar.set_description("train MSE %s" % loss.detach().numpy())

    loss.backward()
    
    optimizer.step()

    monitor.write(model, i, train={"MSELoss": loss.detach()})

    # estimate learned class boundaries
    test_y = model.predict_bst(test_x)
    test_y = test_y.reshape(xx.shape)

plt.clf()
# plot class boundaries
plt.scatter(xx, yy, c=test_y, s=5, alpha=0.6)

# plot training points with true labels
plt.scatter(X[labels == 0][:,0], X[labels == 0][:,1], s=15, marker="o", c="k")
plt.scatter(X[labels == 1][:,0], X[labels == 1][:,1], s=15, marker="^", c="k")

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.title("{} dataset. lr={}; tree depth={}; iter={}".format(DISTR, LR, TREE_DEPTH, i))

plt.savefig(SAVE_DIR + DISTR + "-{}.png".format(i), bbox_inches='tight')

monitor.close()