import numpy as np
import sys

from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from qhoptim.pyt import QHAdam

from src.datasets import Dataset, TorchDataset
from src.LT_models import LTRegressor
from src.losses import NT_Xent
from src.metrics import LT_dendrogram_purity
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.transformations import TransformsSimCLR
from src.utils import deterministic 

DATA_NAME = sys.argv[1]

LR = 0.001
EPOCHS = 100
BATCH_SIZE = 256
SPLIT = 'conv'
COMP = 'none'
TEMP = 1
PROJ_DIM = 64
DROPOUT = 0.
REG = 0.
TREE_DEPTH = 6

pruning = REG > 0

if torch.cuda.is_available():
    pin_memory = True
    device = torch.device("cuda:0")

else:
    pin_memory = False
    device = torch.device("cpu")

print("Training on", device)

data = Dataset(DATA_NAME, normalize=True, seed=459107)
classes = np.unique(data.y_train)
num_classes = max(classes) + 1
in_size = data.X_train.shape[1:]
transform = TransformsSimCLR(in_size)

trainloader = DataLoader(TorchDataset(data.X_train, transform=transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=pin_memory, drop_last=True)
valloader = DataLoader(TorchDataset(data.X_valid, transform=transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=pin_memory, drop_last=True)

test_scores= []
for SEED in [1225, 1337, 2020, 6021991]:

    deterministic(SEED)

    save_dir = Path("./results/constrastive/") / DATA_NAME / "temperature={}/proj-dim={}/depth={}/reg={}/seed={}".format(TEMP, PROJ_DIM, TREE_DEPTH, REG, SEED)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = LTRegressor(TREE_DEPTH, in_size, PROJ_DIM, pruned=pruning, linear=False, split_func=SPLIT, dropout=DROPOUT, COMP_FUNC=COMP)
    model.to(device)

    print(model.count_parameters(), "model's parameters")
    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    # init loss
    criterion = NT_Xent(BATCH_SIZE, TEMP)

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'NT-XENT',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
        'reg': REG,
    }

    best_val_loss = float('inf')
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor, contrastive=True, device=device)

        val_loss = evaluate(valloader, model, {'NT_XENT': criterion}, epoch=e, monitor=monitor, contrastive=True, device=device)

        print("Epoch %i: validation NT_XENT = %f\n" % (e, val_loss['NT_XENT']))

        no_improv += 1
        if val_loss['NT_XENT'] <= best_val_loss:
            best_val_loss = val_loss['NT_XENT']
            best_e = e
            LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, val_NT_XENT=val_loss['NT_XENT'])
            no_improv = 0

        # reduce learning rate if needed
        lr_scheduler.step(val_loss['NT_XENT'])
        monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})

        if no_improv == EPOCHS // 5:
            break

    monitor.close()

    model = LTRegressor.load_model(save_dir)

    score, _ = LT_dendrogram_purity(data.X_test, data.y_test, model, model.latent_tree.bst, num_classes)
    print("Epoch %i: test purity = %f\n" % (best_e, score))
    
    test_scores.append(score)

print(np.mean(test_scores), np.std(test_scores))
np.save(save_dir / '../test-scores.npy', test_scores)
