import numpy as np

from pathlib import Path

from torch.nn import MSELoss
from torch.utils.data import DataLoader

from qhoptim.pyt import QHAdam

from src.LT_models import LTRegressor
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.tabular_datasets import Dataset
from src.utils import make_directory, TorchDataset

SEED = 1225
DATA_NAME = "YEAR"
TREE_DEPTH = 3
REG = 561.7353202746074
MLP_LAYERS = 3
DROPOUT = 0.07600075080048799

LR = 0.01
BATCH_SIZE = 512 
EPOCHS = 1

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

model = LTRegressor(TREE_DEPTH, in_features, out_features, pruned=pruning, linear=LINEAR)

# init optimizer
optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

# init loss
criterion = MSELoss(reduction="sum")

# init train-eval monitoring 
monitor = MonitorTree(pruning, save_dir)

state = {
    'batch-size': BATCH_SIZE,
    'regression': 'linear',
    'loss-function': 'MSE',
    'learning-rate': LR,
    'seed': SEED,
    'bst_depth': TREE_DEPTH,
    'in_size': in_features,
    'out_size': out_features,
    'pruned': pruning,
    'dataset': DATA_NAME,
    'reg': REG,
    'linear': LINEAR
}

best_val_loss = float("inf")
best_e = -1
for e in range(EPOCHS):
    train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)

    val_loss = evaluate(valloader, model, criterion, epoch=e, monitor=monitor)
    print("Epoch %i: validation loss = %f\n" % (e, val_loss))

    if val_loss <= best_val_loss:
        best_val_loss = min(val_loss, best_val_loss)
        best_e = e
        LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, val_loss=val_loss)

monitor.close()
print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss * std ** 2))

model = LTRegressor.load_model(save_dir)
test_loss = evaluate(testloader, model, criterion)
print("test loss (model of epoch {}): {}\n".format(best_e, test_loss * std ** 2))

