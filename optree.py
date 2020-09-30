import sys
import numpy as np

from sklearn.metrics import accuracy_score

from src.baselines import OptTree
from src.datasets import Dataset
from src.utils import deterministic

import time

DATA_NAME = sys.argv[1]
DEPTH = 8

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True)
metric = lambda x, y: 1 - accuracy_score(x, y)

test_losses, train_times, test_times = [], [], []
for SEED in [1225, 1337, 2020, 6021991]:
    
    deterministic(SEED)

    idx = np.random.choice(len(data.X_train), 50_000, replace=False)

    t0 = time.time()
    model = OptTree(DEPTH, data.X_train.shape[1], verbose=True)
    model.train(data.X_train[idx], data.y_train[idx])
    t1 = time.time()
    
    train_times.append(t1 - t0)

    ypred = model.predict(data.X_train)
    loss_train = metric(data.y_train, ypred)

    ypred = model.predict(data.X_valid)
    loss_valid = metric(data.y_valid, ypred)

    t2 = time.time()

    ypred = model.predict(data.X_test)
    loss_test = metric(data.y_test, ypred)
    test_losses.append(loss_test)
    t3 = time.time()

    test_times.append(t3 - t2)

print("Test: %0.5f %0.5f" % (np.mean(test_losses), np.std(test_losses)))
print(f"Train time: {np.mean(train_times)}s")
print(f"Inference time: {np.mean(test_times)}s")

print(f"Tree depth {DEPTH}, num nodes {model.bst.nb_nodes}, num leaves {model.bst.nb_leaves}, num parameters {model.bst.nb_split * (model.in_size + 1)}")
