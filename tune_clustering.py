import numpy as np

from pathlib import Path
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import optuna

from qhoptim.pyt import QHAdam

from src.LT_models import LTClassifier
from src.metrics import LT_dendrogram_purity
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.tabular_datasets import Dataset
from src.utils import make_directory, TorchDataset

SEED = 1337
DATA_NAME = "GLASS"
LR = 0.2
BATCH_SIZE = 128 
EPOCHS = int(1e4)

data = Dataset(DATA_NAME, random_state=SEED, normalize=True)
in_features = data.X_train.shape[1]
classes = np.unique(data.y_train)
num_classes = max(classes) + 1
print(classes)

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, shuffle=True)

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 1, 12)
    REG = trial.suggest_uniform('REG', 0, 1e3)

    save_dir = Path("./results/clustering/") / DATA_NAME / "depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
    make_directory(save_dir)

    pruning = REG > 0

    model = LTClassifier(TREE_DEPTH, in_features, num_classes, pruned=pruning)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))


    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2)

    # init loss
    criterion = CrossEntropyLoss(reduction="sum")

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'CE',
        'learning-rate': LR,
        'seed': SEED,
        'bst_depth': TREE_DEPTH,
        'in_size': in_features,
        'num_classes': num_classes,
        'pruned': pruning,
        'dataset': DATA_NAME,
        'reg': REG,
    }

    best_val_score = 0
    best_e = -1
    for e in tqdm(range(EPOCHS)):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor, prog_bar=False)

        if e % 100 == 0:
            score, _ = LT_dendrogram_purity(data.X_valid, data.y_valid, model, num_classes)
            print("Epoch %i: validation purity = %f\n" % (e, score))

            if score >= best_val_score:
                best_val_score = score
                best_e = e
                LTClassifier.save_model(model, optimizer, state, save_dir, epoch=e, val_dp=score)

            # reduce learning rate if needed
            lr_scheduler.step(score)

    monitor.close()

    return best_val_score

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=DATA_NAME, pruner=optuna.pruners.MedianPruner(), direction="maximize")
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
