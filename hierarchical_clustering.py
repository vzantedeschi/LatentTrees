import numpy as np

from pathlib import Path
from tqdm import tqdm

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

SEED = 1337
DATA_NAME = "COVTYPE"
TREE_DEPTH = 5
REG = 800
LR = 0.2
BATCH_SIZE = 512 
EPOCHS = 20

out_features = [3, 4]
in_features = list(set(range(54)) - set(out_features))

save_dir = Path("./results/clustering-selfsup/") / DATA_NAME / "out-feats=[3,4]/depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
make_directory(save_dir)

pruning = REG > 0

data = Dataset(DATA_NAME, random_state=SEED, normalize=True)
classes = np.unique(data.y_train)
num_classes = max(classes) + 1

data.X_train_in, data.X_valid_in, data.X_test_in = data.X_train[:, in_features], data.X_valid[:, in_features], data.X_test[:, in_features]
data.X_train_out, data.X_valid_out, data.X_test_out = data.X_train[:, out_features], data.X_valid[:, out_features], data.X_test[:, out_features]

trainloader = DataLoader(TorchDataset(data.X_train_in, data.X_train_out), batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid_in, data.X_valid_out), batch_size=BATCH_SIZE*2, shuffle=False)
testloader = DataLoader(TorchDataset(data.X_test_in, data.X_test_out), batch_size=BATCH_SIZE*2, shuffle=False)

model = LTRegressor(TREE_DEPTH, len(in_features), len(out_features), pruned=pruning)

# init optimizer
optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

# init learning rate scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

# init loss
criterion = MSELoss(reduction="sum")

eval_criterion = lambda x, y: (x != y).sum()

# init train-eval monitoring 
monitor = MonitorTree(pruning, save_dir)

state = {
    'batch-size': BATCH_SIZE,
    'loss-function': 'MSE',
    'learning-rate': LR,
    'seed': SEED,
    'bst_depth': TREE_DEPTH,
    'in_size': len(in_features),
    'out_size': len(out_features),
    'pruned': pruning,
    'dataset': DATA_NAME,
    'reg': REG,
    'linear': True,
}

best_val_score = 0
best_e = -1
for e in range(EPOCHS):
    train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=REG, monitor=monitor)

    val_loss = evaluate(valloader, model, criterion, epoch=e, monitor=monitor)
    score, _ = LT_dendrogram_purity(data.X_valid_in, data.y_valid, model, model.latent_tree.bst, num_classes)

    print("Epoch %i: validation loss = %f; validation purity = %f\n" % (e, val_loss, score))

    monitor.write(model, e, val={"Dendrogram Purity": score})

    if score >= best_val_score:
        best_val_score = score
        best_e = e
        LTRegressor.save_model(model, optimizer, state, save_dir, epoch=e, val_loss=val_loss, val_dp=score)

    # reduce learning rate if needed
    lr_scheduler.step(val_loss)
    monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})

monitor.close()
print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss))

model = LTRegressor.load_model(save_dir)
test_loss = evaluate(testloader, model, criterion)
print("test error loss (model of epoch {}): {}\n".format(best_e, test_loss))

score, _ = LT_dendrogram_purity(data.X_test_in, data.y_test, model, model.latent_tree.bst, num_classes)
print("Epoch %i: validation purity = %f\n" % (e, score))