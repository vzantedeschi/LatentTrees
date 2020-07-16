import numpy as np

import torch

from src.trees import BinarySearchTree
from src.monitors import MonitorTree
from src.qp import pruning_qp

# ----------------------------------------------------------------------- LINEAR REGRESSION

class LinearRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(LinearRegression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size, bias=False)     

    def forward(self, x):
        
        return self.linear(x).squeeze()

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size, bias=False)     

    def forward(self, x):

        y_pred = torch.sigmoid(self.linear(x))
        
        return y_pred

# ----------------------------------------------------------------------- LATENT TREES REGRESSION

class LatentTree(torch.nn.Module):

    def __init__(self, bst_depth, dim, pruned=True):

        super(LatentTree, self).__init__()

        self.bst = BinarySearchTree(bst_depth)       
        self.A = torch.nn.Parameter(torch.rand(self.bst.nb_split, dim) * 1e-3)
        
        self.pruned = pruned
        
        if pruned:
            self.eta = torch.nn.Parameter(torch.rand(self.bst.nb_nodes) * 1e-3)


    def forward(self, x):

        q = self._compute_q(x)
        
        if self.pruned:

            self.d = pruning_qp(q, self.eta)
            z = torch.clamp(q, 0, 1)
            z = torch.min(z, self.d)

        else:
            z = torch.clamp(q, 0, 1)

        self.z = z

        return z

    def predict(self, x):

        z = self.forward(x).detach().numpy()

        return self.bst.predict(z)

    def _compute_q(self, x):

        # compute tree paths q
        XA = torch.mm(x, self.A.T)

        q = torch.ones((len(x), self.bst.nb_nodes))

        # upper bound children's q to parent's q        
        # trick to avoid inplace operations involving A
        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes], XA[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes], -XA[:, self.bst.split_nodes])

        for _ in range(self.bst.depth):
            q[:, self.bst.desc_left] = torch.min(q[:, self.bst.desc_left], q[:, self.bst.split_nodes])
            q[:, self.bst.desc_right] = torch.min(q[:, self.bst.desc_right], q[:, self.bst.split_nodes])

        return q

class LTBinaryClassifier(torch.nn.Module):

    def __init__(self, bst_depth, dim, pruned=True):

        super(LTBinaryClassifier, self).__init__()

        # init latent tree optimizer (x -> z)
        self.latent_tree = LatentTree(bst_depth, dim, pruned)

        # init predictor ( [x;z]-> y )
        self.predictor = LogisticRegression(dim + self.latent_tree.bst.nb_nodes, 1)

    def eval(self):
        self.latent_tree.eval()
        self.predictor.eval()

    def forward(self, X):
        
        # add offset
        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        z = self.latent_tree(x)

        xz = torch.cat((x, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.latent_tree.parameters()) + list(self.predictor.parameters())

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return y_pred.detach()

    def predict_bst(self, X):

        # add offset
        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        return self.latent_tree.predict(x)

    def train(self):
        self.latent_tree.train()
        self.predictor.train()

class LTLinearRegressor(torch.nn.Module):

    def __init__(self, bst_depth, in_size, out_size, pruned=True):

        super(LTLinearRegressor, self).__init__()

        # init latent tree optimizer (x -> z)
        self.latent_tree = LatentTree(bst_depth, in_size + 1, pruned)

        # init predictor ( [x;z]-> y )
        self.predictor = LinearRegression(in_size + 1 + self.latent_tree.bst.nb_nodes, out_size)

    def eval(self):
        self.latent_tree.eval()
        self.predictor.eval()

    def forward(self, X):
        
        # add offset
        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        z = self.latent_tree(x)

        xz = torch.cat((x, z), 1)

        return self.predictor(xz)

    def parameters(self):
        return list(self.latent_tree.parameters()) + list(self.predictor.parameters())

    def predict(self, X):

        y_pred = self.forward(X)

        return y_pred.detach()

    def predict_bst(self, X):

        x = torch.cat((X, torch.ones((len(X), 1))), 1)

        return self.latent_tree.predict(x)

    def train(self):
        self.latent_tree.train()
        self.predictor.train()