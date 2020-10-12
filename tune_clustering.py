import numpy as np

from pathlib import Path
from tqdm import tqdm

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import optuna

from qhoptim.pyt import QHAdam

from src.LT_models import LTRegressor
from src.metrics import LT_dendrogram_purity
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

SEED = 1337
DATA_NAME = "DIGITS"
LR = 0.001
EPOCHS = 100

# selecting input and output features for self-supervised training
if DATA_NAME == "COVTYPE":
    out_features = [3, 4]
    in_features = list(set(range(54)) - set(out_features))
    BATCH_SIZE = 512

elif DATA_NAME == "GLASS":
    out_features = [0, 1]
    in_features = list(set(range(9)) - set(out_features))
    BATCH_SIZE = 8

deterministic(SEED)

data = Dataset(DATA_NAME, normalize=True, in_features=in_features, out_features=out_features, flatten=(DATA_NAME == "DIGITS"), seed=459107)
classes = np.unique(data.y_train)
num_classes = max(classes) + 1

root_dir = Path("./results/optuna/clustering-selfsup/") / "{}/out-feats={}/".format(DATA_NAME, out_features)

trainloader = DataLoader(TorchDataset(data.X_train_in, data.X_train_out), batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
valloader = DataLoader(TorchDataset(data.X_valid_in, data.X_valid_out), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=12)

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 2, 6)
    REG = trial.suggest_loguniform('REG', 1e-3, 1e3)
    
    print(f'depth={TREE_DEPTH}, reg={REG}')
    pruning = REG > 0

    save_dir = root_dir / "depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = LTRegressor(TREE_DEPTH, data.X_train_in.shape[1:], data.X_train_out.shape[1:], pruned=pruning)

    print(model.count_parameters(), "model's parameters")

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

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
        'reg': REG,
        'linear': True,
        'in_features': in_features,
        'out_features': out_features,
    }

    best_val_loss = float('inf')
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)
        
        val_loss = evaluate(valloader, model, {'MSE': criterion}, epoch=e, monitor=monitor)
        
        no_improv += 1
        if val_loss['MSE'] <= best_val_loss:
            best_val_loss = val_loss['MSE']
            best_e = e
            LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, val_mse=val_loss['MSE'])
            no_improv = 0

        # reduce learning rate if needed
        lr_scheduler.step(val_loss['MSE'])
        monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})

        if np.isnan(val_loss['MSE']):
            monitor.close()
            raise optuna.TrialPruned()

        if no_improv == EPOCHS // 5:
            break
    
    model = LTRegressor.load_model(save_dir)
    score, _ = LT_dendrogram_purity(data.X_valid_in, data.y_valid, model, model.latent_tree.bst, num_classes)

    print(f"Best model: validation mse = {best_val_loss}; validation purity = {score}\n")

    monitor.write(model, e, val={"Dendrogram Purity": score})

    monitor.close()         

    return score

if __name__ == "__main__":

    study = optuna.create_study(study_name=DATA_NAME, direction="maximize")
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(root_dir / "trials.csv")
