import sys
import numpy as np

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from src.datasets import Dataset
from src.utils import deterministic

import time

DATA_NAME, TREE_DEPTH, MAX_FEATURES, MIN_DECREASE, ALPHA, SPLITTER = sys.argv[1:] 
TREE_DEPTH = int(TREE_DEPTH)
MAX_FEATURES = float(MAX_FEATURES)
MIN_DECREASE = float(MIN_DECREASE)
ALPHA = float(ALPHA)

if DATA_NAME in ["MICROSOFT", "YEAR", "YAHOO"]:
    data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, normalize_target=True)
    print("target mean = %.5f, std = %.5f" % (data.mean_y, data.std_y))
    tree = DecisionTreeRegressor
    metric = lambda x, y: mean_squared_error(x, y) * data.std_y ** 2
else:
    data = Dataset(DATA_NAME, normalize=True, quantile_transform=True)
    tree = DecisionTreeClassifier
    metric = lambda x, y: 1 - accuracy_score(x, y)

test_losses, train_times, test_times = [], [], [] 
for SEED in [1225, 1337, 2020, 6021991]:
    deterministic(SEED)

    t0 = time.time()
    model = tree(splitter=SPLITTER, max_depth=TREE_DEPTH, max_features=MAX_FEATURES, min_impurity_decrease=MIN_DECREASE, ccp_alpha=ALPHA)
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

    test_losses.append(loss_test)
    train_times.append(t1-t0)
    test_times.append(t3-t2)

print(f"Tree max depth {model.get_depth()}, num nodes {model.tree_.node_count}, num leaves {model.get_n_leaves()}, num parameters {sum(model.tree_.feature > -1) * 2}")
print(np.mean(test_losses), np.std(test_losses))
print("AVg train time", np.mean(train_times))
print("AVg test time", np.mean(test_times))

