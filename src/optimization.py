import torch

from tqdm import tqdm

from src.LT_models import LTBinaryClassifier
from src.monitors import MonitorTree

# Train LT Binary Classifier with gradient descent on full dataset
def train_batch(x, y, bst_depth=2, nb_iter=1e4, lr=5e-1, reg=10, norm="inf"):

    n, d = x.shape

    pruning = reg > 0
    
    model = LTBinaryClassifier(bst_depth, d + 1, pruned=pruning)
    monitor = MonitorTree(pruning, "runs/norm={}/reg={}/".format(norm, reg))

    # init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # init loss
    criterion = torch.nn.BCELoss(reduction="mean")

    # cast to pytorch Tensors
    t_y = torch.from_numpy(y[:, None]).float()
    t_x = torch.from_numpy(x).float()

    model.train()

    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:

        # print(model.latent_tree.eta.detach().numpy())
        optimizer.zero_grad()

        y_pred = model(t_x)

        bce = criterion(y_pred, t_y)
        if pruning:
            loss = bce + reg * torch.norm(model.latent_tree.eta, p=norm)
            pbar.set_description("train BCE + reg %s" % loss.detach().numpy())

        else:
            loss = bce
            pbar.set_description("train BCE %s" % loss.detach().numpy())

        loss.backward()
        
        optimizer.step()

        monitor.write(model, i, train={"BCELoss": bce.detach()})

    monitor.close()

    return model