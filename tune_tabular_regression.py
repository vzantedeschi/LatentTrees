import numpy as np

from pathlib import Path

from torch.nn import MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import optuna

from src.LT_models import LTLinearRegressor
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.tabular_datasets import Dataset
from src.utils import make_directory, TorchDataset

SEED = 1337
DATA_NAME = "MICROSOFT"
LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 20

# load dataset with same configuration as in https://github.com/Qwicen/node/blob/master/notebooks/year_node_shallow.ipynb
data = Dataset(DATA_NAME, random_state=SEED, quantile_transform=True, quantile_noise=1e-3, normalize=True)
in_features = data.X_train.shape[1]
out_features = 1

# normalize y
mu, std = data.y_train.mean(), data.y_train.std()
normalize = lambda x: ((x - mu) / std).astype(np.float32)
data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])

print("mean = %.5f, std = %.5f" % (mu, std))

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, num_workers=12, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, num_workers=12, shuffle=False)

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 1, 12)
    REG = trial.suggest_uniform('REG', 0, 1e5)

    pruning = REG > 0
    save_dir = Path("./results/optuna/") / DATA_NAME / "depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
    make_directory(save_dir)
    model = LTLinearRegressor(TREE_DEPTH, in_features, out_features, pruned=pruning)

    # init optimizer
    optimizer = SGD(model.parameters(), lr=LR)

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    # init loss
    criterion = MSELoss(reduction="sum")

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'regression': 'linear',
        'loss-function': 'MSE',
        'learning-rate': LR,
        'seed': SEED,
        'tree-depth': TREE_DEPTH,
        'dataset': DATA_NAME,
        'reg': REG,
    }

    best_val_loss = float("inf")
    best_e = -1
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)

        val_loss = evaluate(valloader, model, criterion, epoch=e, monitor=monitor)
        print("Epoch %i: validation loss = %f\n" % (e, val_loss))

        if val_loss <= best_val_loss:
            best_val_loss = min(val_loss, best_val_loss)
            best_e = e
            # save_model(model, optimizer, state, save_dir)
        
        # reduce learning rate if needed
        lr_scheduler.step(val_loss)

        trial.report(val_loss * std**2, e)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            monitor.close()
            raise optuna.TrialPruned()

    monitor.close()
    print("DEPTH={}, REG={}, best validation loss (epoch {}): {}, unscaled {}\n".format(TREE_DEPTH, REG, best_e, best_val_loss, best_val_loss * std ** 2))

    return best_val_loss * std**2

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=DATA_NAME, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)

