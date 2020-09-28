import sys

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from src.datasets import Dataset
from src.utils import deterministic

import optuna

DATA_NAME = sys.argv[1]

SEED = 1337
deterministic(SEED)

if DATA_NAME in ["MICROSOFT", "YEAR", "YAHOO"]:
    data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, normalize_target=True)
    print("target mean = %.5f, std = %.5f" % (data.mean_y, data.std_y))
    tree = DecisionTreeRegressor
    metric = lambda x, y: mean_squared_error(x, y) * data.std_y ** 2
else:
    data = Dataset(DATA_NAME, normalize=True, quantile_transform=True)
    tree = DecisionTreeClassifier
    metric = lambda x, y: 1 - accuracy_score(x, y)

root_dir = f"results/cart/optuna/{DATA_NAME}/seed={SEED}/"

deterministic(SEED)

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 2, 10)
    MAX_FEATURES = trial.suggest_uniform("MAX_FEATURES", 0.5, 1)
    MIN_DECREASE = trial.suggest_uniform("MIN_DECREASE", 0, 1)
    ALPHA = trial.suggest_loguniform("ALPHA", 1e-16, 1e2)
    SPLITTER = trial.suggest_categorical("SPLITTER", ["best", "random"])

    model = tree(splitter=SPLITTER, max_depth=TREE_DEPTH, max_features=MAX_FEATURES, min_impurity_decrease=MIN_DECREASE, ccp_alpha=ALPHA)

    model = model.fit(data.X_train, data.y_train)

    ypred = model.predict(data.X_train)
    loss_train = metric(data.y_train, ypred)

    print(model.get_params())
    print("Train:", loss_train)

    ypred = model.predict(data.X_valid)
    loss_valid = metric(data.y_valid, ypred)
    print("Validation:", loss_valid)

    return loss_valid

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=f"cart-{DATA_NAME}")
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(Path(root_dir) / 'trials.csv')
