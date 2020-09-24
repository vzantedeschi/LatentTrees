import numpy as np

import node
import torch, torch.nn as nn
import torch.nn.functional as F

from qhoptim.pyt import QHAdam
from tqdm import tqdm

from pathlib import Path

import optuna

from src.datasets import Dataset
from src.utils import deterministic

DATA_NAME = "YEAR"
BATCH_SIZE = 512
EPOCHS = 100
SEED = 1337
LR = 0.001

device = torch.device("cpu")

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, normalize_target=True)
in_features = data.X_train.shape[1]
out_features = 1
print("target mean = %.5f, std = %.5f" % (data.mean_y, data.std_y))

root_dir = f"results/node/optuna/{DATA_NAME}/seed={SEED}/"

deterministic(SEED)

def objective(trial):

    TREE_DEPTH = trial.suggest_categorical('TREE_DEPTH', [6, 8])
    NUM_TREES = trial.suggest_categorical('NUM_TREES', [1024, 2048])
    NUM_LAYERS = trial.suggest_categorical('NUM_LAYERS', [2, 4, 8])
    TREE_DIM = trial.suggest_categorical('TREE_DIM', [2, 3])

    trial_name = root_dir + f"depth={TREE_DEPTH}/tree={NUM_TREES}/layers={NUM_LAYERS}/tree-dim={TREE_DIM}"
    print("trial saved in", trial_name)

    model = nn.Sequential(
        node.DenseBlock(in_features, NUM_TREES // NUM_LAYERS, num_layers=NUM_LAYERS, tree_dim=TREE_DIM, depth=TREE_DEPTH, flatten_output=False, choice_function=node.entmax15, bin_function=node.entmoid15),
        node.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
    )

    with torch.no_grad():
        res = model(torch.as_tensor(data.X_train[:5000]))
        # trigger data-aware init

    optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }

    trainer = node.Trainer(
        model=model, 
        loss_function=F.mse_loss,
        experiment_name=trial_name,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        verbose=False,
        n_last_checkpoints=5
    )

    best_mse = float('inf')
    best_step_mse = 0
    report_frequency = 100
    early_stopping_rounds = 5000

    for batch in tqdm(node.iterate_minibatches(data.X_train.float(), data.y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS), desc=""):
        metrics = trainer.train_on_batch(*batch, device=device)
        
        loss_history.append(metrics['loss'])

        if trainer.step % report_frequency == 0:
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')
            mse = trainer.evaluate_mse(
                data.X_valid.float(), data.y_valid, device=device, batch_size=BATCH_SIZE*2)

            if mse < best_mse:
                best_mse = mse
                best_step_mse = trainer.step
                trainer.save_checkpoint(tag='best_mse')
            
            trainer.load_checkpoint()  # last
            trainer.remove_old_temp_checkpoints()

            trial.report(mse * data.std_y**2, trainer.step)

            # Handle pruning based on the intermediate value.
            if trial.should_prune() or np.isnan(mse):
                
                raise optuna.TrialPruned()

        if trainer.step > best_step_mse + early_stopping_rounds:          
            break

    print("Best step: ", best_step_mse)
    print("Best Val MSE: %0.5f" % (best_mse))

    return best_mse * data.std_y**2

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=f"node-{DATA_NAME}", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(Path(root_dir) / 'trials.csv')
