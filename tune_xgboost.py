import xgboost as xgb
import numpy as np

import sys
import optuna

from pathlib import Path

from src.datasets import Dataset
from src.utils import deterministic

DATA_NAME = sys.argv[1]
WORKERS = int(sys.argv[2])

ROUNDS = 5000
SEED = 1337

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

root_dir = Path(f"results/xgboost/optuna/{DATA_NAME}/seed={SEED}/")
root_dir.mkdir(parents=True, exist_ok=True)

deterministic(SEED)

def objective(trial):

    ETA = trial.suggest_loguniform("ETA", 1e-7, 1)
    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 2, 10)
    SUBSAMPLE = trial.suggest_uniform("SUBSAMPLE", 0.5, 1)
    BYTREE = trial.suggest_uniform("BYTREE", 0.5, 1)
    BYLEVEL = trial.suggest_uniform("BYLEVEL", 0.5, 1)
    CHILD_WEIGHT = trial.suggest_loguniform("CHILD_WEIGHT", 1e-16, 1e5)
    ALPHA = trial.suggest_loguniform("ALPHA", 1e-16, 1e2)
    LAMBDA = trial.suggest_loguniform("LAMBDA", 1e-16, 1e2)
    GAMMA = trial.suggest_loguniform("GAMMA", 1e-16, 1e2)

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

    print(param)

    evallist = [(dvalid, 'valid')]

    bst = xgb.train(param, dtrain, ROUNDS, evallist, early_stopping_rounds=20)

    ypred = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)

    if DATA_NAME in ["MICROSOFT", "YEAR", "YAHOO"]:

        best_loss = np.mean((data.y_valid - ypred) ** 2) * data.std_y**2

    else:
        ypred[ypred < 0.5] = 0
        ypred[ypred >= 0.5] = 1.
        
        best_loss = (data.y_valid != ypred).mean()

    print("Best step: ", bst.best_ntree_limit)
    print("Best Val Loss: %0.5f" % (best_loss))

    return best_loss 

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=f"node-{DATA_NAME}")
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(Path(root_dir) / 'trials.csv')
