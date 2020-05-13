import numpy as np

from tqdm import tqdm

import torch

from src.trees import BinarySearchTree

# ----------------------------------------------------------------------- LINEAR REGRESSION

class LinearRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(LinearRegression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size, bias=False)     

    def forward(self, x):
        
        return self.linear(x)

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size, bias=False)     

    def forward(self, x):

        y_pred = torch.sigmoid(self.linear(x))
        
        return y_pred

# ----------------------------------------------------------------------- LATENT TREES REGRESSION

class LPSparseMAP(torch.nn.Module):

    def __init__(self, bst_depth, dim):

        super(LPSparseMAP, self).__init__()

        self.bst = BinarySearchTree(bst_depth)       
        self.A = torch.nn.Parameter(torch.rand(self.bst.nb_split, dim))

    def forward(self, x):

        q = self._compute_q(x)

        # non differentiable output for q > 1
        z = torch.clamp(q, 0, 1)

        return z

    def predict(self, x):

        z = self.forward(x).detach().numpy()

        return self.bst.predict(z)

    def _compute_q(self, x):

        # compute tree paths q
        XA = torch.mm(x, self.A.T)

        q = torch.ones((len(x), self.bst.nb_nodes))
        
        # trick to avoid inplace operations involving A
        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes], XA[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes], -XA[:, self.bst.split_nodes])

        # upper bound children's q to parent's q
        for _ in range(self.bst.depth):
            q[:, self.bst.desc_left] = torch.min(q[:, self.bst.desc_left], q[:, self.bst.split_nodes])
            q[:, self.bst.desc_right] = torch.min(q[:, self.bst.desc_right], q[:, self.bst.split_nodes])

        return q

class BinaryClassifier(torch.nn.Module):

    def __init__(self, bst_depth, dim):

        super(BinaryClassifier, self).__init__()

        # init latent tree optimizer (x -> z)
        self.sparseMAP = LPSparseMAP(bst_depth, dim)

        # init predictor ( [x;z]-> y )
        self.predictor = LogisticRegression(dim + self.sparseMAP.bst.nb_nodes, 1)

    def eval(self):
        self.sparseMAP.eval()
        self.predictor.eval()

    def forward(self, X):
        
        # add offset
        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        z = self.sparseMAP(x)

        xz = torch.cat((x, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.sparseMAP.parameters()) + list(self.predictor.parameters())

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return y_pred.detach()

    def predict_bst(self, X):

        # add offset
        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        return self.sparseMAP.predict(x)

    def train(self):
        self.sparseMAP.train()
        self.predictor.train()

class LinearRegressor(torch.nn.Module):

    def __init__(self, bst_depth, in_size1, in_size2, out_size):

        super(LinearRegressor, self).__init__()

        # init latent tree optimizer (x2 -> z)
        self.sparseMAP = LPSparseMAP(bst_depth, in_size2 + 1)

        # init predictor ( [x1;z]-> y )
        self.predictor = LinearRegression(in_size1 + 1 + self.sparseMAP.bst.nb_nodes, out_size)

    def eval(self):
        self.sparseMAP.eval()
        self.predictor.eval()

    def forward(self, X1, X2):
        
        # add offset
        x1 = torch.cat((X1, torch.ones((len(X1), 1))), 1)
        x2 = torch.cat((X2, torch.ones((len(X2), 1))), 1)

        z = self.sparseMAP(x2)

        xz = torch.cat((x1, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.sparseMAP.parameters()) + list(self.predictor.parameters())

    def predict(self, X1, X2):

        y_pred = self.forward(X1, X2)

        return y_pred.detach()

    def predict_bst(self, X2):

        x2 = torch.cat((X2, torch.ones((len(X2), 1))), 1)

        return self.sparseMAP.predict(x2)

    def train(self):
        self.sparseMAP.train()
        self.predictor.train()

def train_batch(x, y, bst_depth=2, nb_iter=1e4, lr=5e-1):

    n, d = x.shape

    model = BinaryClassifier(bst_depth, d + 1)

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

        optimizer.zero_grad()

        y_pred = model(t_x)

        loss = criterion(y_pred, t_y)

        loss.backward()
        
        optimizer.step()

        pbar.set_description("BCE train loss %s" % loss.detach().numpy())

    return model