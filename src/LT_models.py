import numpy as np
import torch

from pathlib import Path

from src.trees import BinarySearchTree
from src.monitors import MonitorTree
from src.qp import pruning_qp

# ----------------------------------------------------------------------- LINEAR REGRESSION

class Linear(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(Regression, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size, bias=False)     

    def forward(self, x):
        
        return self.linear(x)

class MLP(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(MLP, self).__init__()

        self.net = torch.nn.Sequential(
                torch.nn.Linear(in_size, in_size*2),
                torch.nn.ELU(),
                torch.nn.Linear(in_size*2, in_size),
                torch.nn.ELU(),
                torch.nn.Linear(in_size, out_size),
            )

    def forward(self, x):
        
        return self.net(x)

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
        self.A = torch.nn.Parameter(torch.rand(self.bst.nb_split, dim))
        
        self.pruned = pruned
        
        if pruned:
            self.eta = torch.nn.Parameter(torch.rand(self.bst.nb_nodes))

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

class LTRegressor(torch.nn.Module):

    def __init__(self, bst_depth, in_size, out_size, pruned=True, linear=True):

        super(LTRegressor, self).__init__()

        # init latent tree optimizer (x -> z)
        self.latent_tree = LatentTree(bst_depth, in_size + 1, pruned)

        # init predictor ( [x;z]-> y )
        if linear:
            self.predictor = Linear(in_size + 1 + self.latent_tree.bst.nb_nodes, out_size)
        else:
            self.predictor = MLP(in_size + 1 + self.latent_tree.bst.nb_nodes, out_size)

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

    @staticmethod
    def load_model(load_dir, **kwargs):

        checkpoint = torch.load(Path(load_dir) / 'model.t7')
        
        model = LTRegressor(checkpoint['bst_depth'], checkpoint['in_size'], checkpoint['out_size'], checkpoint['pruned'], checkpoint['linear'])
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer' in kwargs.keys():
            kwargs['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

        return model
  
    def save_model(self, optimizer, state, save_dir, **kwargs):

        try:
            state_dict = self.module.state_dict()

        except AttributeError:
            state_dict = self.state_dict()

        state['model_state_dict'] = state_dict
        state['optimizer_state_dict'] = optimizer.state_dict()
        state.update(kwargs)

        torch.save(state, Path(save_dir) / 'model.t7')
