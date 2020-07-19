import numpy as np
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

def train_stochastic(dataloader, model, optimizer, criterion, epoch, reg=1, norm=float("inf"), monitor=None):

    model.train()

    last_iter = epoch * len(dataloader)

    train_obj = 0.
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):

        optimizer.zero_grad()

        t_x, t_y = batch

        y_pred = model(t_x)

        loss = criterion(y_pred, t_y)

        if reg > 0:

            obj = loss + reg * torch.norm(model.latent_tree.eta, p=norm)
            train_obj += obj.detach().numpy()

            pbar.set_description("avg train loss + reg %f" % (train_obj / (i + 1)))

        else:

            obj = loss
            train_obj += obj.detach().numpy()

            pbar.set_description("avg train loss %f" % (train_obj / (i + 1)))

        obj.backward()

        optimizer.step()

        if monitor:
            monitor.write(model, i + last_iter, train={"Loss": loss.detach()})

def evaluate(dataloader, model, criterion, epoch=None, monitor=None):

    model.eval()

    total_loss = 0.
    
    for i, batch in enumerate(dataloader):

        t_x, t_y = batch

        y_pred = model(t_x)

        loss = criterion(y_pred, t_y)
        total_loss += loss.detach()

    if monitor:
        monitor.write(model, epoch, val={"Loss": total_loss / i})

    return total_loss.numpy() / i
