import numpy as np

from pathlib import Path

from torch.nn import BCELoss
from torch.utils.data import DataLoader

from qhoptim.pyt import QHAdam

from src.LT_models import LTBinaryClassifier
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

import time
import sys

DATA_NAME = sys.argv[1]
TREE_DEPTH = int(sys.argv[2])
REG = float(sys.argv[3])
MLP_LAYERS = int(sys.argv[4])
DROPOUT = float(sys.argv[5])

LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 100

pruning = REG > 0

data = Dataset(DATA_NAME, normalize=True, seed=459107)
print('classes', np.unique(data.y_test))

test_losses, train_times, test_times = [], [], []
for SEED in [1225, 1337, 2020, 6021991]:
    deterministic(SEED)

    save_dir = Path("./results/tabular-quantile/") / DATA_NAME / "depth={}/reg={}/mlp-layers={}/dropout={}/seed={}".format(TREE_DEPTH, REG, MLP_LAYERS, DROPOUT, SEED)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, num_workers=16, shuffle=False)
    testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=BATCH_SIZE*2, num_workers=16, shuffle=False)

    model = LTBinaryClassifier(TREE_DEPTH, data.X_train.shape[1], reg=REG)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init loss
    loss = BCELoss(reduction="sum")
    criterion = lambda x, y: loss(x.float(), y.float())

    # evaluation criterion => error rate
    eval_criterion = lambda x, y: (x != y).sum()

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'BCE',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
        'reg': REG,
    }

    best_val_loss = float("inf")
    best_e = -1
    no_improv = 0
    t0 = time.time()
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, monitor=monitor)

        val_loss = evaluate(valloader, model, {'ER': eval_criterion}, epoch=e, monitor=monitor)
        print("Epoch %i: validation loss = %f\n" % (e, val_loss["ER"]))
        no_improv += 1

        if val_loss["ER"] < best_val_loss:
            best_val_loss = val_loss["ER"]
            best_e = e
            no_improv = 0
            LTBinaryClassifier.save_model(model, optimizer, state, save_dir, epoch=e, val_er=best_val_loss)

        if no_improv == EPOCHS // 5:
            break
    t1 = time.time()
    monitor.close()
    print("best validation error rate (epoch {}): {}\n".format(best_e, best_val_loss))

    model = LTBinaryClassifier.load_model(save_dir)
    t2 = time.time()
    test_loss = evaluate(testloader, model, {'ER': eval_criterion})
    print("test error rate (model of epoch {}): {}\n".format(best_e, test_loss['ER']))
    t3 = time.time()
    test_losses.append(test_loss['ER'])
    train_times.append(t1 - t0)
    test_times.append(t3 - t2)

print(np.mean(test_losses), np.std(test_losses))
np.save(save_dir / '../test-losses.npy', test_losses)
print("Avg train time", np.mean(train_times))
print("Avg test time", np.mean(test_times))
