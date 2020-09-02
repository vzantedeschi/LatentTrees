import numpy as np

from pathlib import Path
from tqdm import tqdm

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from qhoptim.pyt import QHAdam

from src.LT_models import LTRegressor
from src.metrics import LT_dendrogram_purity
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.tabular_datasets import Dataset
from src.utils import make_directory, TorchDataset

DATA_NAME = "GLASS"
TREE_DEPTH = 8
REG = 498.073
LR = 0.1
EPOCHS = 100

pruning = REG > 0

if torch.cuda.is_available():
    pin_memory = True
    device = torch.device("cuda:0")

else:
    pin_memory = False
    device = torch.device("cpu")

print("Training on", device)

# selecting input and output features for self-supervised training
if DATA_NAME == "ALOI":
    in_features = [0, 2] # R and B
    out_features = [1] # G
    BATCH_SIZE = 128
    SPLIT = 'conv'

elif DATA_NAME == "COVTYPE":
    out_features = [3, 4]
    in_features = list(set(range(54)) - set(out_features))
    BATCH_SIZE = 512
    SPLIT = 'linear'

elif DATA_NAME == "GLASS":
    out_features = [7, 8]
    in_features = list(set(range(9)) - set(out_features))
    BATCH_SIZE = 8
    SPLIT = 'linear'

test_scores= []
for SEED in [1225, 1337, 2020, 6021991]:

    data = Dataset(DATA_NAME, random_state=SEED, normalize=True)
    classes = np.unique(data.y_train)
    num_classes = max(classes) + 1

    save_dir = Path("./results/clustering-selfsup/") / DATA_NAME / "out-feats={}/depth={}/reg={}/seed={}".format(out_features, TREE_DEPTH, REG, SEED)

    save_dir.mkdir(parents=True, exist_ok=True)

    data.X_train_in, data.X_valid_in, data.X_test_in = data.X_train[:, in_features], data.X_valid[:, in_features], data.X_test[:, in_features]
    data.X_train_out, data.X_valid_out, data.X_test_out = data.X_train[:, out_features], data.X_valid[:, out_features], data.X_test[:, out_features]
    
    if DATA_NAME == "ALOI":
        trainloader = DataLoader(TorchDataset(data.X_train_in, data.X_train_out, means=(data.mean[in_features], data.mean[out_features]), stds=(data.std[in_features], data.std[out_features])), batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=pin_memory)
        valloader = DataLoader(TorchDataset(data.X_valid_in, data.X_valid_out, means=(data.mean[in_features], data.mean[out_features]), stds=(data.std[in_features], data.std[out_features])), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=12, pin_memory=pin_memory)
        testloader = DataLoader(TorchDataset(data.X_test_in, data.X_test_out, means=(data.mean[in_features], data.mean[out_features]), stds=(data.std[in_features], data.std[out_features])), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=12, pin_memory=pin_memory)

    else:
        trainloader = DataLoader(TorchDataset(data.X_train_in, data.X_train_out), batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=pin_memory)
        valloader = DataLoader(TorchDataset(data.X_valid_in, data.X_valid_out), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=12, pin_memory=pin_memory)
        testloader = DataLoader(TorchDataset(data.X_test_in, data.X_test_out), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=12, pin_memory=pin_memory)

    model = LTRegressor(TREE_DEPTH, data.X_train_in.shape[1:], data.X_train_out.shape[1:], pruned=pruning, split_func=SPLIT)

    print(model.count_parameters(), "model's parameters")
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
        'reg': REG,
        'linear': True,
    }

    best_val_loss = float('inf')
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)

        val_loss = evaluate(valloader, model, {'MSE': criterion}, epoch=e, monitor=monitor)

        print("Epoch %i: validation mse = %f\n" % (e, val_loss['MSE']))

        no_improv += 1
        if val_loss['MSE'] <= best_val_loss:
            best_val_loss = val_loss['MSE']
            best_e = e
            LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, val_mse=val_loss['MSE'])
            no_improv = 0

        # reduce learning rate if needed
        lr_scheduler.step(val_loss['MSE'])
        monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})

        if no_improv == EPOCHS // 5:
            break

    monitor.close()

    model = LTRegressor.load_model(save_dir)

    score, _ = LT_dendrogram_purity(data.X_test_in, data.y_test, model, model.latent_tree.bst, num_classes)
    print("Epoch %i: test purity = %f\n" % (best_e, score))
    
    test_scores.append(score)

print(np.mean(test_scores), np.std(test_scores))
np.save(save_dir / '../test-scores.npy', test_scores)
