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

SEED = 1337
DATA_NAME = "HIGGS"
TREE_DEPTH = 5
REG = 0
LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 100

save_dir = Path("./results/tab-datasets/") / DATA_NAME / "depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
save_dir.mkdir(parents=True, exist_ok=True)

pruning = REG > 0

data = Dataset(DATA_NAME, normalize=True, seed=459107)
print('classes', np.unique(data.y_test))

deterministic(SEED)

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, shuffle=False)
testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=BATCH_SIZE*2, shuffle=False)

model = LTBinaryClassifier(TREE_DEPTH, data.X_train.shape[1], pruned=pruning)

# init optimizer
optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

# init loss
criterion = BCELoss(reduction="sum")

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
for e in range(EPOCHS):
    train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)

    val_loss = evaluate(valloader, model, eval_criterion, epoch=e, monitor=monitor)
    print("Epoch %i: validation loss = %f\n" % (e, val_loss))

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_e = e
        LTBinaryClassifier.save_model(model, optimizer, state, save_dir, epoch=e, val_loss=val_loss)

monitor.close()
print("best validation error rate (epoch {}): {}\n".format(best_e, best_val_loss))

model = LTBinaryClassifier.load_model(save_dir)
test_loss = evaluate(testloader, model, eval_criterion)
print("test error rate (model of epoch {}): {}\n".format(best_e, test_loss))

