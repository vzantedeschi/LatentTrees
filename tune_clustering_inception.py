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
from src.transformations import TransformInception
from src.utils import deterministic

SEED = 1337
DATA_NAME = "ALOI"
LR = 0.001
EPOCHS = 100
BATCH_SIZE = 512
NB_FEATS = 100

out_features = list(range(NB_FEATS))
in_features = list(range(NB_FEATS, 1000))

deterministic(SEED)

data = Dataset(DATA_NAME, seed=459107)
classes = np.unique(data.y_train)
num_classes = max(classes) + 1

root_dir = Path("./results/optuna/clustering-inception/") / f"{DATA_NAME}/out-feats={NB_FEATS}/"

transform = TransformInception(in_features, out_features)

trainloader = DataLoader(TorchDataset(data.X_train, transform=transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valloader = DataLoader(TorchDataset(data.X_valid, transform=transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 8, 10)
    REG = trial.suggest_loguniform('REG', 1e-3, 1e3)
    
    print(f'depth={TREE_DEPTH}, reg={REG}')
    pruning = REG > 0

    save_dir = root_dir / "depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = LTRegressor(TREE_DEPTH, len(in_features), NB_FEATS, pruned=pruning)

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
        'transform': "inception",
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
    score, _ = LT_dendrogram_purity(valloader, data.y_valid, model, model.latent_tree.bst, num_classes)

    print(f"Best model: validation mse = {best_val_loss}; validation purity = {score}\n")

    monitor.write(model, e, val={"Dendrogram Purity": score})

    monitor.close()         

    return score

if __name__ == "__main__":

    study = optuna.create_study(study_name=DATA_NAME + "-inception", direction="maximize")
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(root_dir / "trials.csv")
