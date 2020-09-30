import time
import sys

import numpy as np
import xgboost as xgb

from src.datasets import Dataset
from src.utils import deterministic

DATA_NAME, WORKERS, ETA, TREE_DEPTH, SUBSAMPLE, BYTREE, BYLEVEL, CHILD_WEIGHT, ALPHA, LAMBDA, GAMMA = sys.argv[1:]

ETA, TREE_DEPTH, SUBSAMPLE, BYTREE, BYLEVEL, CHILD_WEIGHT, ALPHA, LAMBDA, GAMMA = float(ETA), int(TREE_DEPTH), float(SUBSAMPLE), float(BYTREE), float(BYLEVEL), float(CHILD_WEIGHT), float(ALPHA), float(GAMMA), float(GAMMA)

ROUNDS = 5000

if DATA_NAME in ["MICROSOFT", "YEAR", "YAHOO"]:
    data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, normalize_target=True)
    obj = 'reg:squarederror'
    metric = 'rmse'

else:
    data = Dataset(DATA_NAME, normalize=True, quantile_transform=True)
    obj = 'reg:logistic'
    metric = 'error'

dtrain = xgb.DMatrix(data.X_train, label=data.y_train)
dvalid = xgb.DMatrix(data.X_valid, label=data.y_valid)
dtest = xgb.DMatrix(data.X_test, label=data.y_test)

param = {
    'objective': obj,
    'nthread': WORKERS,
    'eval_metric': metric,
    'eta': ETA,
    'gamma': GAMMA,
    'max_depth': TREE_DEPTH,
    'min_child_weight': CHILD_WEIGHT,
    'subsample': SUBSAMPLE,
    'colsample_bytree': BYTREE,
    'colsample_bylevel': BYLEVEL,
    'lambda': LAMBDA,
    'alpha': ALPHA,
}

evallist = [(dvalid, 'eval')]

test_losses, train_times, test_times = [], [], [] 
for SEED in [1225, 1337, 2020, 6021991]:

    param['seed'] = SEED
    deterministic(SEED)
    t0 = time.time()

    bst = xgb.train(param, dtrain, ROUNDS, evallist, early_stopping_rounds=20)

    t1 = time.time()
    print(f"Training time: {t1 - t0}s")

    ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    if DATA_NAME in ["MICROSOFT", "YEAR", "YAHOO"]:

        test_loss = np.mean((data.y_test - ypred) ** 2) * data.std_y**2

    else:
        ypred[ypred < 0.5] = 0
        ypred[ypred >= 0.5] = 1.
        
        test_loss = (data.y_test != ypred).mean()

    t2 = time.time()

    print('Best step: ', bst.best_ntree_limit)
    print("Test loss: %0.5f" % (test_loss))

    print(f"Inference time: {t2 - t1}s")

    test_losses.append(test_loss)
    train_times.append(t1 - t0)
    test_times.append(t2 - t1)

print(np.mean(test_losses), np.std(test_losses))
print("AVg train time", np.mean(train_times))
print("AVg test time", np.mean(test_times))