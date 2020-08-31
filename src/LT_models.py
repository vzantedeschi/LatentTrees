import numpy as np
import torch

from pathlib import Path

from src.trees import BinarySearchTree
from src.monitors import MonitorTree
from src.qp import pruning_qp

# ----------------------------------------------------------------------- REGRESSION

class Linear(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(Linear, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size)     

    def forward(self, x):
        
        return self.linear(x)

class MLP(torch.nn.Module):
    
    def __init__(self, in_size, out_size, layers=1, dropout=0., **kwargs):
        
        super(MLP, self).__init__()
        
        if layers == 1:
            units = [(in_size, out_size)]
        else:
            units = [(in_size, 64)]
            for i in range(1, layers - 1):
                units.append((units[-1][1], units[-1][1] * 2))
            units.append((units[-1][1], out_size))
        
        self.layers = []
        for i, u in enumerate(units):
            self.layers.append(torch.nn.Linear(*u))
            
            if i < layers - 1: # end the model with a linear layer
                self.layers.append(torch.nn.ELU())
                
                if dropout > 0.:
                    self.layers.append(torch.nn.Dropout(dropout))
        
        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        
        return self.net(x)

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size, linear, **kwargs):
        
        super(LogisticRegression, self).__init__()

        if linear:
            self.net = Linear(in_size, out_size)
        else:
            self.net = MLP(in_size, out_size, **kwargs)   

    def forward(self, x):

        y_pred = torch.sigmoid(self.net(x))
        
        return y_pred

# ----------------------------------------------------------------------- LATENT TREE LAYER

class LatentTree(torch.nn.Module):

    def __init__(self, bst_depth, dim, pruned=True, split_func='linear'):

        super(LatentTree, self).__init__()

        self.in_size = dim
        self.bst = BinarySearchTree(bst_depth)      

        self.split_func = split_func
        if split_func == 'linear':
            self.split = torch.nn.Linear(dim, self.bst.nb_split)

        elif split_func == 'elu':
            self.split = torch.nn.Sequential(
                    torch.nn.Linear(dim, self.bst.nb_split),
                    torch.nn.ELU()
                )
        elif split_func == 'conv':
            self.split = torch.nn.Sequential(
                    torch.nn.Conv2d(dim, 16, 3, stride=2),
                    torch.nn.Conv2d(16, 32, 3, stride=2),
                    torch.nn.ELU(),
                    torch.nn.Conv2d(32, 16, 3, stride=2),
                    torch.nn.Conv2d(16, 8, 3, stride=2),
                    torch.nn.ELU(),
                    torch.nn.Conv2d(8, self.bst.nb_split, 4, stride=4),
                    torch.nn.MaxPool2d((2, 4)),
                    torch.nn.Flatten(),
                )

        else:
            raise NotImplementedError
        
        self.pruned = pruned
        
        if pruned:
            self.eta = torch.nn.Parameter(torch.rand(self.bst.nb_nodes))

    def forward(self, x):

        q = self._compute_q(x)
        z = torch.clamp(q, 0, 1)

        if self.pruned:

            self.d = pruning_qp(q, self.eta)
            clamped_d = torch.clamp(self.d, 0, 1)
            z = torch.min(z, clamped_d)

        self.z = z

        return z

    def predict(self, x):

        z = self.forward(x).detach().numpy()

        return z, self.bst.predict(z)

    def _compute_q(self, x):

        # compute tree paths q
        XA = self.split(x)
        q = torch.ones((len(x), self.bst.nb_nodes))

        # upper bound children's q to parent's q        
        # trick to avoid inplace operations involving A
        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes], XA[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes], -XA[:, self.bst.split_nodes])

        for _ in range(self.bst.depth):
            q[:, self.bst.desc_left] = torch.min(q[:, self.bst.desc_left], q[:, self.bst.split_nodes])
            q[:, self.bst.desc_right] = torch.min(q[:, self.bst.desc_right], q[:, self.bst.split_nodes])

        return q

# ------------------------------------------------------------------------------- NN MODELS

class LTModel(torch.nn.Module):

    def __init__(self):

        torch.nn.Module.__init__(self)

    def train(self):
        self.latent_tree.train()
        self.predictor.train()

    def eval(self):
        self.latent_tree.eval()
        self.predictor.eval() 

    def parameters(self):
        return list(self.latent_tree.parameters()) + list(self.predictor.parameters())

    def forward(self, X):

        z = self.latent_tree(X)

        xz = torch.cat((X, z), 1)

        return self.predictor(xz)

    def predict_bst(self, X):

        return self.latent_tree.predict(X)

    def db_distance(self, X):

        if self.latent_tree.split_func == 'linear':

            XA = self.latent_tree.split(X)
            return (XA / torch.norm(self.latent_tree.split.weight, dim=1, p=2))
        
        else:
            raise NotImplementedError

    @classmethod
    def load_model(cls, load_dir, **kwargs):

        checkpoint = torch.load(Path(load_dir) / 'model.t7')

        model = cls(**checkpoint)
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
        state.update(self.params)
        state.update(kwargs)

        torch.save(state, Path(save_dir) / 'model.t7')

class LTBinaryClassifier(LTModel):

    def __init__(self, bst_depth, in_size, pruned=True, linear=True, **kwargs):

        super(LTBinaryClassifier, self).__init__()

        self.params = locals()
        del self.params['self']

        # init latent tree optimizer (x -> z)
        if 'split_func' in kwargs:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned, split_func=kwargs['split_func'])
        else:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned)

        # init predictor ( [x;z]-> y )
        self.predictor = LogisticRegression(in_size + self.latent_tree.bst.nb_nodes, 1, linear, **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return y_pred.detach()

class LTClassifier(LTModel):

    def __init__(self, bst_depth, in_size, num_classes, pruned=True, linear=True, **kwargs):

        super(LTClassifier, self).__init__()

        self.params = locals()
        del self.params['self']

        # init latent tree optimizer (x -> z)
        if 'split_func' in kwargs:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned, split_func=kwargs['split_func'])
        else:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned)

        # init predictor ( [x;z]-> y )
        if linear:
            self.predictor = Linear(in_size + self.latent_tree.bst.nb_nodes, num_classes)
        else:
            self.predictor = MLP(in_size + self.latent_tree.bst.nb_nodes, num_classes, **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred = torch.argmax(y_pred, axis=1)

        return y_pred.detach()

class LTRegressor(LTModel):

    def __init__(self, bst_depth, in_size, out_size, pruned=True, linear=True, **kwargs):

        super(LTRegressor, self).__init__()

        self.params = locals()
        del self.params['self']

        # init latent tree optimizer (x -> z)
        if 'split_func' in kwargs:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned, split_func=kwargs['split_func'])
        else:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned)

        # init predictor ( [x;z]-> y )
        if linear:
            self.predictor = Linear(in_size + self.latent_tree.bst.nb_nodes, out_size)
        else:
            self.predictor = MLP(in_size + self.latent_tree.bst.nb_nodes, out_size, **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)

        return y_pred.detach()