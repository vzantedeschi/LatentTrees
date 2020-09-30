import numpy as np
import time
import sys

from pathlib import Path

from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from qhoptim.pyt import QHAdam

from src.baselines import NeuralDecisionForest
from src.optimization import train_ndf, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

DATA_NAME = sys.argv[1]
WORKERS = int(sys.argv[2])
DEPTH = 6
FEAT_RATE = 0.847924044521007
JOINT = True
NUM_TREES = 1

LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 100

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True)
in_features = data.X_train.shape[1]
num_classes = max(data.y_train) + 1

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, num_workers=WORKERS, shuffle=False)
testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=BATCH_SIZE*2, num_workers=WORKERS, shuffle=False)

test_losses, train_times, test_times = [], [], []
for SEED in [1225, 1337, 2020, 6021991]:

    save_dir = Path("./results/ndf/tabular-quantile/") / DATA_NAME / f"trees={NUM_TREES}/depth={DEPTH}/feat-rate={FEAT_RATE}/joint={JOINT}/seed={SEED}"
    save_dir.mkdir(parents=True, exist_ok=True)

    deterministic(SEED)

    model = NeuralDecisionForest(in_features, num_classes, num_trees=NUM_TREES, tree_depth=DEPTH, tree_feature_rate=FEAT_RATE, jointly_training=JOINT)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))
    
    # evaluation criterion => error rate
    eval_criterion = lambda x, y: (x != y).sum()

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'NLL',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
    }

    t0 = time.time()
    best_val_loss = float("inf")
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_ndf(trainloader, model, optimizer, epoch=e, jointly_training=JOINT)

        val_loss = evaluate(valloader, model, {'valid_ER': eval_criterion}, epoch=e)
        print(f"Epoch {e}: {val_loss}\n")
        
        no_improv += 1
        if val_loss['valid_ER'] <= best_val_loss:
            best_val_loss = val_loss['valid_ER']
            best_e = e
            NeuralDecisionForest.save_model(model, optimizer, state, save_dir, epoch=e, **val_loss)
            no_improv = 0

        # reduce learning rate if needed
        lr_scheduler.step(val_loss['valid_ER'])

        if no_improv == EPOCHS // 5:
            break

    t1 = time.time()
    print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss))

    model = NeuralDecisionForest.load_model(save_dir)

    t2 = time.time()
    test_loss = evaluate(testloader, model, {'test_ER': eval_criterion})
    print("test loss (model of epoch {}): {}\n".format(best_e, test_loss['test_ER']))

    t3 = time.time()
    test_losses.append(test_loss['test_ER'])
    train_times.append(t1 - t0)
    test_times.append(t3 - t2)

print(np.mean(test_losses), np.std(test_losses))
print("Training time", np.mean(train_times))
print("Inference time", np.mean(test_times))
print("num parameters", model.count_parameters())
