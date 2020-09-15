import numpy as np

from pathlib import Path

from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import optuna

from qhoptim.pyt import QHAdam

from src.LT_models import LTRegressor
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

SEED = 1225
DATA_NAME = "MICROSOFT"
LR = 0.01
BATCH_SIZE = 512 
EPOCHS = 100
LINEAR = False

data = Dataset(DATA_NAME, normalize=True, normalize_target=True)
in_features = data.X_train.shape[1]
out_features = 1
print("target mean = %.5f, std = %.5f" % (data.mean_y, data.std_y))

root_dir = Path("./results/optuna/tabular/") / "{}/linear={}/".format(DATA_NAME, LINEAR)

trainloader = DataLoader(TorchDataset((data.X_train, data.y_train)), batch_size=BATCH_SIZE, num_workers=12, shuffle=True)
valloader = DataLoader(TorchDataset((data.X_valid, data.y_valid)), batch_size=BATCH_SIZE*2, num_workers=12, shuffle=False)

deterministic(SEED)

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 2, 6)
    REG = trial.suggest_loguniform('REG', 1e-3, 1e3)
    MLP_LAYERS = trial.suggest_int('MLP_LAYERS', 2, 7)
    DROPOUT = trial.suggest_uniform('DROPOUT', 0.0, 0.5)
    SPLIT_FUNC = trial.suggest_categorical("SPLIT_FUNC", ["linear", "elu"])

    COMP_FUNC = trial.suggest_categorical('COMP_FUNC', ["none", "concatenate"])
    if COMP_FUNC == "concatenate":
        TWO_PHASED = trial.suggest_categorical("TWO_PHASED", [True, False])
    else:
        TWO_PHASED = False

    pruning = REG > 0
    save_dir = root_dir / "comp={}/twophased={}/split={}/depth={}/reg={}/mlp-layers={}/dropout={}/seed={}".format(COMP_FUNC, TWO_PHASED, SPLIT_FUNC, TREE_DEPTH, REG, MLP_LAYERS, DROPOUT, SEED)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("trial saved in", save_dir)
    
    model = LTRegressor(TREE_DEPTH, in_features, out_features, pruned=pruning, linear=LINEAR, layers=MLP_LAYERS, dropout=DROPOUT, split_func=SPLIT_FUNC, comp_func=COMP_FUNC)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    # init loss
    criterion = MSELoss(reduction="sum")

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'MSE',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
    }

    if TWO_PHASED:
        model.freeze("skip")

    best_val_loss = float("inf")
    best_e = -1
    for e in range(EPOCHS):

        if TWO_PHASED and e == EPOCHS // 2:
            model.unfreeze("skip")

        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)
            
        val_loss = evaluate(valloader, model, {'MSE': criterion}, epoch=e, monitor=monitor)

        if val_loss["MSE"] <= best_val_loss:
            best_val_loss = val_loss["MSE"]
            best_e = e
            # save_model(model, optimizer, state, save_dir)
        
        # reduce learning rate if needed
        lr_scheduler.step(val_loss["MSE"])

        trial.report(val_loss["MSE"] * data.std_y**2, e)

        # Handle pruning based on the intermediate value.
        if trial.should_prune() or np.isnan(val_loss["MSE"]):
            monitor.close()
            raise optuna.TrialPruned()

    monitor.close()

    return best_val_loss * data.std_y**2

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=DATA_NAME, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(root_dir / 'trials.csv')

