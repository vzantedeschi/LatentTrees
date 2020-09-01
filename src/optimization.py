import numpy as np
import torch

from tqdm import tqdm

def train_batch(x, y, LT_model, optimizer, criterion, nb_iter=1e4, reg=10, norm=float("inf"), monitor=None):

    n, d = x.shape

    pruning = reg > 0
    
    # cast to pytorch Tensors
    t_y = torch.from_numpy(y[:, None]).float()
    t_x = torch.from_numpy(x).float()

    LT_model.train()

    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:

        # print(LT_model.latent_tree.eta.detach().numpy())
        optimizer.zero_grad()

        y_pred = LT_model(t_x)

        loss = criterion(y_pred, t_y)
        if pruning:
            obj = loss + reg * torch.norm(LT_model.latent_tree.eta, p=norm)
            pbar.set_description("train loss + reg %s" % obj.detach().numpy())

        else:
            obj = loss
            pbar.set_description("train loss %s" % loss.detach().numpy())

        obj.backward()
        
        optimizer.step()

        monitor.write(LT_model, i, train={"Loss": loss.detach()})

def train_stochastic(dataloader, model, optimizer, criterion, epoch, reg=1, norm=float("inf"), monitor=None, prog_bar=True):

    model.train()

    last_iter = epoch * len(dataloader)

    train_obj = 0.

    if prog_bar:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader

    for i, batch in enumerate(pbar):

        optimizer.zero_grad()

        t_x, t_y = batch

        if t_y.dim() > 2: # predictors support only flatten output atm
            t_y = t_y.view(len(t_y), -1)

        y_pred = model(t_x).squeeze()
        
        loss = criterion(y_pred, t_y.float()) / len(t_x)

        if reg > 0:

            obj = loss + reg * model.latent_tree.bst.nb_nodes * torch.norm(model.latent_tree.eta, p=norm)
            train_obj += obj.detach().numpy()

            if prog_bar:
                pbar.set_description("avg train loss + reg %f" % (train_obj / (i + 1)))

        else:

            obj = loss
            train_obj += obj.detach().numpy()

            if prog_bar:
                pbar.set_description("avg train loss %f" % (train_obj / (i + 1)))

        obj.backward()

        optimizer.step()

        if monitor:
            monitor.write(model, i + last_iter, train={"Loss": loss.detach()})
            
def evaluate(dataloader, model, criteria, epoch=None, monitor=None):

    model.eval()

    total_losses = {k: 0. for k in criteria.keys()}
    
    num_points = 0
    for batch in dataloader:

        t_x, t_y = batch
        
        if t_y.dim() > 2: # predictors support only flatten output atm
            t_y = t_y.view(len(t_y), -1)
        
        num_points += len(t_x)

        y_pred = model.predict(t_x).squeeze()

        for k in criteria.keys():
            loss = criteria[k](y_pred, t_y)
            total_losses[k] += loss.detach()

    if monitor:
        monitor.write(model, epoch, val={k: loss / num_points for k, loss in total_losses.items()})

    return {k: loss.numpy() / num_points for k, loss in total_losses.items()}
