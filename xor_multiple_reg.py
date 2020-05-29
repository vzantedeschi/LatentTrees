import numpy as np
import torch

from src.datasets import toy_dataset
from src.optimization import train_batch

DISTR = "xor"
N = 1000
TREE_DEPTH = 4
LR = 0.1
ITER = 1e4

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# generate toy dataset
X, Y = toy_dataset(N, DISTR)

for reg in [10**i for i in range(-2, 2)]:
    for norm in [0, 1, float('inf')]:

        # train latent class tree and logistic regressor
        model = train_batch(X, Y, bst_depth=TREE_DEPTH, nb_iter=ITER, lr=LR, reg=reg, norm=norm)
