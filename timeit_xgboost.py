import time
import xgboost as xgb

from src.datasets import Dataset

DATA_NAME = "YAHOO"
EPOCHS = 100

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, normalize_target=True)

dtrain = xgb.DMatrix(data.X_train, label=data.y_train)
dvalid = xgb.DMatrix(data.X_valid, label=data.y_valid)
dtest = xgb.DMatrix(data.X_test, label=data.y_test)

t0 = time.time()
param = {'objective': 'reg:squarederror'}
param['nthread'] = 16
param['eval_metric'] = 'rmse'

evallist = [(dvalid, 'eval'), (dtrain, 'train')]

bst = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=EPOCHS // 5)

t1 = time.time()
print(f"Training time: {t1 - t0}s")

ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

test_mse = np.mean((data.y_test - ypred) ** 2)
t2 = time.time()

print('Best step: ', bst.best_ntree_limit)
print("Test MSE: %0.5f" % (test_mse * data.std_y ** 2))

print(f"Inference time: {t2 - t1}s")
