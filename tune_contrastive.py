import numpy as np

from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR
from torch.utils.data import DataLoader, RandomSampler

import optuna

from torchlars import LARS

from src.datasets import Dataset, TorchDataset
from src.LT_models import LTRegressor
from src.losses import NT_Xent
from src.metrics import LT_dendrogram_purity
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.transformations import TransformsSimCLR
from src.utils import deterministic 

DATA_NAME = "ALOI"

SEED = 1337
PROJ_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 200
SPLIT = 'conv'
COMP = 'none'

LR = 0.6 * BATCH_SIZE / 256
WU_LR = LR / 4 ** 5

WD = 1e-6

pin_memory = False
device = torch.device("cpu")

print("Training on", device)

data = Dataset(DATA_NAME, seed=459107)
classes = np.unique(data.y_train)
num_classes = max(classes) + 1
in_size = data.X_train.shape[1:]
transform = TransformsSimCLR(in_size)

train_dataset = TorchDataset(data.X_train, transform=transform)
# to augment dataset
train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=100*BATCH_SIZE)

trainloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=16, pin_memory=pin_memory)
valloader = DataLoader(TorchDataset(data.X_valid, transform=transform, test=True), batch_size=BATCH_SIZE, num_workers=16, pin_memory=pin_memory)

deterministic(SEED)

root_dir = Path("./results/optuna/contrastive/") / f"{DATA_NAME}/proj={PROJ_DIM}/comp={COMP}/split={SPLIT}"

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 6, 10)
    REG = trial.suggest_loguniform('REG', 1e-3, 1e3)
    TEMP = trial.suggest_uniform('TEMP', 0, 1)
    DROPOUT = trial.suggest_uniform('DROPOUT', 0, 0.5)
    
    print(f'depth={TREE_DEPTH}, reg={REG}, temp={TEMP}, dropout={DROPOUT}')
    pruning = REG > 0

    save_dir = root_dir / f"depth={TREE_DEPTH}/reg={REG}/temp={TEMP}/dropout={DROPOUT}/seed={SEED}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = LTRegressor(TREE_DEPTH, in_size, PROJ_DIM, pruned=pruning, linear=False, split_func=SPLIT, dropout=DROPOUT, COMP_FUNC=COMP)
    model.to(device)

    print(model.count_parameters(), "model's parameters")

    # init optimizer
    optimizer = LARS(SGD(model.parameters(), lr=WU_LR, weight_decay=WD))
    
    # init learning rate schedulers
    lmbda = lambda epoch: 4
    wu_scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=1e-8, last_epoch=4)

    # init loss
    criterion = NT_Xent(TEMP)

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'NT-XENT',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
        'reg': REG,
        'temperatur': TEMP,
    }

    best_val_loss = float('inf')
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor, contrastive=True, device=device)

        val_loss = evaluate(valloader, model, {'NT_XENT': criterion}, epoch=e, monitor=monitor, contrastive=True, device=device)

        no_improv += 1
        if val_loss['NT_XENT'] <= best_val_loss:
            best_val_loss = val_loss['NT_XENT']
            best_e = e
            LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, val_NT_XENT=val_loss['NT_XENT'])
            no_improv = 0
        
        if e < 5:
            wu_scheduler.step()
        else:
            lr_scheduler.step()

        monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})

        if np.isnan(val_loss['NT_XENT']):
            monitor.close()
            raise optuna.TrialPruned()

        if no_improv == EPOCHS // 5:
            break
    
    model = LTRegressor.load_model(save_dir)

    score, _ = LT_dendrogram_purity(data.X_valid, data.y_valid, model, model.latent_tree.bst, num_classes)

    print(f"Best model, epoch {best_e}: validation mse = {best_val_loss}; validation purity = {score}\n")

    monitor.write(model, e, val={"Dendrogram Purity": score})

    monitor.close()         

    return score

if __name__ == "__main__":

    study = optuna.create_study(study_name=DATA_NAME, direction="maximize")
    study.optimize(objective, n_trials=10)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(root_dir / "trials.csv")
