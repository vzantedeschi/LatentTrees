import numpy as np

from pathlib import Path

from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from qhoptim.pyt import QHAdam

from src.LT_models import LTRegressor
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.tabular_datasets import Dataset
from src.utils import make_directory, TorchDataset

DATA_NAME = "MICROSOFT"
LINEAR = False
TREE_DEPTH=8 
REG=784.2480977010307
MLP_LAYERS=3
DROPOUT=0.10054922066470592 

LR = 0.01
BATCH_SIZE = 512 
EPOCHS = 100

test_losses = []
for SEED in [1225, 1337, 2020, 6021991]:
    save_dir = Path("./results/tabular/") / DATA_NAME / "depth={}/reg={}/mlp-layers={}/dropout={}/seed={}".format(TREE_DEPTH, REG, MLP_LAYERS, DROPOUT, SEED)
    make_directory(save_dir)

    pruning = REG > 0

    data = Dataset(DATA_NAME, random_state=SEED, normalize=True)
    in_features = data.X_train.shape[1]
    out_features = 1

    # normalize after applying other transformations
    mu, std = data.y_train.mean(), data.y_train.std()
    normalize = lambda x: ((x - mu) / std).astype(np.float32)
    data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])

    print("mean = %.5f, std = %.5f" % (mu, std))

    trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, shuffle=False)
    testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=BATCH_SIZE*2, shuffle=False)

    model = LTRegressor(TREE_DEPTH, in_features, out_features, pruned=pruning, linear=LINEAR, layers=MLP_LAYERS, dropout=DROPOUT)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init loss
    criterion = MSELoss(reduction="sum")

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
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)

        val_loss = evaluate(valloader, model, {'valid_MSE': criterion}, epoch=e, monitor=monitor)
        print(f"Epoch {e}: {val_loss}\n")
        
        no_improv += 1
        if val_loss['valid_MSE'] <= best_val_loss:
            best_val_loss = val_loss['valid_MSE']
            best_e = e
            LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, **val_loss)
            no_improv = 0

        # reduce learning rate if needed
        lr_scheduler.step(val_loss['valid_MSE'])

        if no_improv == EPOCHS // 4:
            break

    monitor.close()
    print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss * std ** 2))

    model = LTRegressor.load_model(save_dir)
    test_loss = evaluate(testloader, model, {'test_MSE': criterion})
    print("test loss (model of epoch {}): {}\n".format(best_e, test_loss['test_MSE'] * std ** 2))

    test_losses.append(test_loss['test_MSE'] * std ** 2)

print(np.mean(test_losses), np.std(test_losses))
np.save(save_dir / '../test-losses.npy', test_losses)

