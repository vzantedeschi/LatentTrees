import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from src.datasets import Dataset
from src.utils import deterministic

import time

DATA_NAME = sys.argv[1]

SEED = 1337
deterministic(SEED)

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, normalize_target=True)
in_features = data.X_train.shape[1]
out_features = 1
print("target mean = %.5f, std = %.5f" % (data.mean_y, data.std_y))

t0 = time.time()
model = DecisionTreeRegressor()
model = model.fit(data.X_train, data.y_train)
t1 = time.time()

print(f"Training time: {t1 - t0}s")

ypred = model.predict(data.X_valid)
loss_valid = mean_squared_error(data.y_valid, ypred)
print("Validation loss:", loss_valid * data.std_y ** 2)

t2 = time.time()

ypred = model.predict(data.X_test)
loss_test = mean_squared_error(data.y_test, ypred)

t3 = time.time()
print("Test MSE: %0.5f" % (loss_test * data.std_y ** 2))
print(f"Inference time: {t3 - t2}s")

print(f"Tree max depth {model.get_depth()}, num nodes {model.tree_.node_count}, num leaves {model.get_n_leaves()}, num parameters {sum(model.tree_.feature > -1) * 2}")
