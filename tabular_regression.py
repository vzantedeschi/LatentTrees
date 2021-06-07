import numpy as np
import time

from pathlib import Path

from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from qhoptim.pyt import QHAdam

from src.LT_models import LTRegressor
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

import sys

LINEAR = False
DATA_NAME = sys.argv[1]
TREE_DEPTH = int(sys.argv[2])
REG = float(sys.argv[3])
MLP_LAYERS = int(sys.argv[4])
DROPOUT = float(sys.argv[5])

LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 100

pruning = REG > 0

data = Dataset(DATA_NAME, normalize=True, normalize_target=True)
in_features = data.X_train.shape[1]
out_features = 1
print("target mean = %.5f, std = %.5f" % (data.mean_y, data.std_y))

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, num_workers=16, shuffle=False)
testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=BATCH_SIZE*2, num_workers=16, shuffle=False)

test_losses, train_time, test_time = [], [], []

for SEED in [1225, 1337, 2020, 6021991]:
    save_dir = Path("./results/tabular/") / DATA_NAME / "depth={}/reg={}/mlp-layers={}/dropout={}/seed={}".format(TREE_DEPTH, REG, MLP_LAYERS, DROPOUT, SEED)
    save_dir.mkdir(parents=True, exist_ok=True)

    deterministic(SEED)

    model = LTRegressor(TREE_DEPTH, in_features, out_features, reg=REG, linear=LINEAR, layers=MLP_LAYERS, dropout=DROPOUT)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init loss
    loss = MSELoss(reduction="sum")
    criterion = lambda x, y: loss(x.float(), y.float())

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'MSE',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
    }

    best_val_loss = float("inf")
    best_e = -1
    no_improv = 0
    t0 = time.time()
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, monitor=monitor)

        val_loss = evaluate(valloader, model, {'valid_MSE': criterion}, epoch=e, monitor=monitor)
        print(f"Epoch {e}: {val_loss}\n")
        
        no_improv += 1
        if val_loss['valid_MSE'] < best_val_loss:
            best_val_loss = val_loss['valid_MSE']
            best_e = e
            LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, **val_loss)
            no_improv = 0

        # reduce learning rate if needed
        lr_scheduler.step(val_loss['valid_MSE'])

        if no_improv == EPOCHS // 4:
            break

    t1 = time.time()

    monitor.close()
    print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss * data.std_y ** 2))

    model = LTRegressor.load_model(save_dir)
    t2 = time.time()
    test_loss = evaluate(testloader, model, {'test_MSE': criterion})
    print("test loss (model of epoch {}): {}\n".format(best_e, test_loss['test_MSE'] * data.std_y ** 2))
    t3 = time.time()
    test_losses.append(test_loss['test_MSE'] * data.std_y ** 2)
    test_time.append(t3 - t2)
    train_time.append(t1 - t0)

print("Number of parameters", model.count_parameters())
print(np.mean(test_losses), np.std(test_losses))
print("Train time", np.mean(train_time))
print("Test time", np.mean(test_time))
np.save(save_dir / '../test-losses.npy', test_losses)

