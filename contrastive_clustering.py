import numpy as np
import sys

from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR
from torch.utils.data import DataLoader, RandomSampler

from torchlars import LARS

from src.datasets import Dataset, TorchDataset
from src.LT_models import LTRegressor
from src.losses import NT_Xent
from src.metrics import LT_dendrogram_purity
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.transformations import TransformsSimCLR
from src.utils import deterministic 

DATA_NAME = sys.argv[1]

if DATA_NAME == "ALOI":
    PROJ_DIM = 32
    TREE_DEPTH = 6
else:
    PROJ_DIM = 8
    TREE_DEPTH = 4

BATCH_SIZE = 1024
EPOCHS = 100
SPLIT = 'conv'
COMP = 'none'
TEMP = 0.5
DROPOUT = 0.
REG = 0.

LR = 0.6 * BATCH_SIZE / 256
WU_LR = LR / 4 ** 5

WD = 1e-6
pruning = REG > 0

if torch.cuda.is_available():
    pin_memory = True
    device = torch.device("cuda:0")

else:
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

trainloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=6, pin_memory=pin_memory)
valloader = DataLoader(TorchDataset(data.X_valid, transform=transform, test=True), batch_size=BATCH_SIZE, num_workers=6, pin_memory=pin_memory)

test_scores= []
for SEED in [1225]:

    deterministic(SEED)

    save_dir = Path("./results/constrastive/") / DATA_NAME / "temperature={}/proj-dim={}/depth={}/reg={}/seed={}".format(TEMP, PROJ_DIM, TREE_DEPTH, REG, SEED)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = LTRegressor(TREE_DEPTH, in_size, PROJ_DIM, pruned=pruning, linear=False, split_func=SPLIT, dropout=DROPOUT, COMP_FUNC=COMP)
    model.to(device)

    print(model.count_parameters(), "model's parameters")
    # init optimizer
    optimizer = LARS(SGD(model.parameters(), lr=WU_LR, weight_decay=WD))
    
    # init learning rate schedulers
    lmbda = lambda epoch: 4
    wu_scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=1e-8, last_epoch=9)

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
    }

    best_val_loss = float('inf')
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor, contrastive=True, device=device)

        val_loss = evaluate(valloader, model, {'NT_XENT': criterion}, epoch=e, monitor=monitor, contrastive=True, device=device)
        print("Epoch %i: validation NT_XENT = %f\n" % (e, val_loss['NT_XENT']))

        model.cpu()
        val_score, hist = LT_dendrogram_purity(data.X_valid, data.y_valid, model, model.latent_tree.bst, num_classes)
        print("Epoch %i: validation purity = %f" % (e, val_score))
        model.to(device)

        no_improv += 1
        if val_loss['NT_XENT'] <= best_val_loss:
            best_val_loss = val_loss['NT_XENT']
            best_e = e
            LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, val_NT_XENT=val_loss['NT_XENT'], val_dp=val_score)
            no_improv = 0
        
        if e < 5:
            wu_scheduler.step()
        else:
            lr_scheduler.step()

        monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})
        monitor.write(model, e, val={"dp": val_score})

        del val_score, hist
        if no_improv == EPOCHS // 5:
            break

    monitor.close()

    model = LTRegressor.load_model(save_dir)

    score, _ = LT_dendrogram_purity(data.X_test, data.y_test, model, model.latent_tree.bst, num_classes)
    print("Epoch %i: test purity = %f\n" % (best_e, score))
    
    test_scores.append(score)

print(np.mean(test_scores), np.std(test_scores))
np.save(save_dir / '../test-scores.npy', test_scores)
