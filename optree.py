import sys

from sklearn.metrics import accuracy_score

from src.baselines import OptTree
from src.datasets import Dataset
from src.utils import deterministic

import time

DATA_NAME = sys.argv[1]
DEPTH = 8

SEED = 1337
deterministic(SEED)

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True)
metric = lambda x, y: 1 - accuracy_score(x, y)

t0 = time.time()
model = OptTree(DEPTH, data.X_train.shape[1])
model = model.fit(data.X_train, data.y_train)
t1 = time.time()

print(f"Training time: {t1 - t0}s")

ypred = model.predict(data.X_train)
loss_train = metric(data.y_train, ypred)
print("Train:", loss_train)

ypred = model.predict(data.X_valid)
loss_valid = metric(data.y_valid, ypred)
print("Validation:", loss_valid)

t2 = time.time()

ypred = model.predict(data.X_test)
loss_test = metric(data.y_test, ypred)

t3 = time.time()
print("Test: %0.5f" % (loss_test))
print(f"Inference time: {t3 - t2}s")

print(f"Tree depth {DEPTH}, num nodes {model.tree_.node_count}, num leaves {model.get_n_leaves()}, num parameters {sum(model.tree_.feature > -1) * 2}")
