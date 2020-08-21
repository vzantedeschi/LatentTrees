import numpy as np

from pathlib import Path
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from qhoptim.pyt import QHAdam

from src.LT_models import LTClassifier
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.tabular_datasets import Dataset
from src.utils import make_directory, TorchDataset

SEED = 1337
DATA_NAME = "GLASS"
TREE_DEPTH = 4
REG = 1
LR = 0.2
BATCH_SIZE = 128 
EPOCHS = int(5e4)

save_dir = Path("./results/clustering/") / DATA_NAME / "depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
make_directory(save_dir)

pruning = REG > 0

data = Dataset(DATA_NAME, random_state=SEED, normalize=True)
in_features = data.X_train.shape[1]
classes = np.unique(data.y_train)
num_classes = max(classes) + 1
print(classes)

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, shuffle=False)
testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=BATCH_SIZE*2, shuffle=False)

model = LTClassifier(TREE_DEPTH, in_features, num_classes, pruned=pruning)

# init optimizer
optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

# init learning rate scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

# init loss
criterion = CrossEntropyLoss(reduction="sum")

# evaluation criterion => error rate
eval_criterion = lambda x, y: (x != y).sum()

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

best_val_loss = float("inf")
best_e = -1
for e in tqdm(range(EPOCHS)):
    train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor, prog_bar=False)

    if e % 100 == 0:
        val_loss = evaluate(valloader, model, eval_criterion, epoch=e, monitor=monitor)
        print("Epoch %i: validation loss = %f\n" % (e, val_loss))

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_e = e
            LTClassifier.save_model(model, optimizer, state, save_dir, epoch=e, val_loss=val_loss)

        # reduce learning rate if needed
        lr_scheduler.step(val_loss)

monitor.close()
print("best validation error rate (epoch {}): {}\n".format(best_e, best_val_loss))

model = LTClassifier.load_model(save_dir)
test_loss = evaluate(testloader, model, eval_criterion)
print("test error rate (model of epoch {}): {}\n".format(best_e, test_loss))

