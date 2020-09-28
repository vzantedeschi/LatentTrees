import numpy as np

from pathlib import Path

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import optuna

from qhoptim.pyt import QHAdam

from src.baselines import DNDT
from src.optimization import train_stochastic, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

SEED = 1225
DATA_NAME = "CLICK"
LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 100
LINEAR = False

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, seed=459107)
print('classes', np.unique(data.y_test))

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, num_workers=30, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, num_workers=30, shuffle=False)

root_dir = Path("./results/dntn/optuna/") / DATA_NAME

deterministic(SEED)

def objective(trial):

    TEMP = trial.suggest_uniform('TEMP', 0, 1)
    CUTS = trial.suggest_categorical('CUTS', [1, 2])
    print(f'TEMP={TEMP}, CUTS={CUTS}')

    model = DNDT(data.X_train.shape[1], 2, CUTS, TEMP)

    print(model.count_parameters(), "model's parameters")
    
    save_dir = root_dir / f"cuts={CUTS}/temp={TEMP}/"
    save_dir.mkdir(parents=True, exist_ok=True)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    # init loss
    criterion = CrossEntropyLoss(reduction="sum")

    # evaluation criterion => error rate
    eval_criterion = lambda x, y: (x.long() != y.long()).sum()

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'CE',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
    }

    best_val_loss = float("inf")
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=0)

        val_loss = evaluate(valloader, model, {'ER': eval_criterion}, epoch=e)
        
        no_improv += 1
        if val_loss['ER'] <= best_val_loss:
            best_val_loss = val_loss['ER']
            best_e = e
            no_improv = 0
            # save_model(model, optimizer, state, save_dir)
        
        # reduce learning rate if needed
        lr_scheduler.step(val_loss['ER'])

        trial.report(val_loss['ER'], e)
        # Handle pruning based on the intermediate value.
        if trial.should_prune() or np.isnan(val_loss['ER']):
            raise optuna.TrialPruned()

        if no_improv == 10:
            break

    print("Best validation ER:", best_val_loss)

    return best_val_loss

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=DATA_NAME, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(root_dir / 'trials.csv')
