import numpy as np
import torch

from pathlib import Path

from src.trees import BinarySearchTree
from src.monitors import MonitorTree
from src.qp import pruning_qp
from src.utils import concat_func, freezed_concat_func, none_func

# ----------------------------------------------------------------------- REGRESSION

class Linear(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        
        super(Linear, self).__init__()

        self.linear = torch.nn.Linear(in_size, out_size)     

    def forward(self, x):
        
        return self.linear(x)

class MLP(torch.nn.Module):
    
    def __init__(self, in_size, out_size, layers=2, dropout=0., **kwargs):
        
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

        self.bst = BinarySearchTree(bst_depth)      

        self.split_func = split_func
        self.bias = torch.nn.Parameter(torch.zeros(self.bst.nb_split), requires_grad=True)
        self.bias_init = False

        if split_func == 'linear':
            
            self.in_size = np.prod(dim)

            self.split = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(self.in_size, self.bst.nb_split)
                )

        elif split_func == 'elu':

            self.in_size = np.prod(dim)

            self.split = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(self.in_size, self.bst.nb_split),
                    torch.nn.ELU()
                )
        else:
            raise NotImplementedError
        
        self.pruned = pruned
        
        if pruned:
            self.eta = torch.nn.Parameter(torch.rand(self.bst.nb_nodes) * 2 - 1)

    def forward(self, x):

        if not self.bias_init:
            self._init_bias(x)
            self.bias_init = True

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
        s = self.split(x) + self.bias
        act_s = self.act(s)

        q = torch.ones((len(x), self.bst.nb_nodes), device=x.device)

        # upper bound children's q to parent's q        
        # trick to avoid inplace operations involving A
        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes], act_s[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes], -act_s[:, self.bst.split_nodes])

        for _ in range(self.bst.depth):
            q[:, self.bst.desc_left] = torch.min(q[:, self.bst.desc_left], q[:, self.bst.split_nodes])
            q[:, self.bst.desc_right] = torch.min(q[:, self.bst.desc_right], q[:, self.bst.split_nodes])

        return q

    def _init_bias(self, x):

        s = self.split(x)
        bias = -s.mean(0)

        node_masks = [np.array([True] * len(x))] # set of points assigned to each node
        for l in range(1, self.bst.nb_split, 2): # loop over left nodes

            r = l + 1 # right node
            p = self.bst.parent(l) # parent node

            s_p = (s[:, p] + bias[p]).detach().numpy() # parent's projections

            node_masks.append(node_masks[p] & (s_p > 0)) # points going to the left node
            node_masks.append(node_masks[p] & (s_p < 0)) # points going to the rigth node

            bias[l] = -s[node_masks[l], l].mean()
            bias[r] = -s[node_masks[r], r].mean()

        self.bias = torch.nn.parameter.Parameter(bias, requires_grad=True)

# ------------------------------------------------------------------------------- NN MODELS

class LTModel(torch.nn.Module):

    def __init__(self, bst_depth, in_size, pruned, **kwargs):

        torch.nn.Module.__init__(self)

        # init latent tree optimizer (x -> z)
        if 'split_func' in kwargs:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned, split_func=kwargs['split_func'])
        else:
            self.latent_tree = LatentTree(bst_depth, in_size, pruned)

        # init composition function for constructing input for predictor
        if 'comp_func' in kwargs and kwargs['comp_func'] != "concatenate":

            if kwargs['comp_func'] == "none":
                self.comp = none_func
                self.pred_in_size = self.latent_tree.bst.nb_leaves # predictor's input size

            else:
                raise NotImplementedError

        else:
            self.comp = concat_func
            self.pred_in_size = np.prod(in_size) + self.latent_tree.bst.nb_leaves # predictor's input size

    def train(self):
        self.latent_tree.train()
        self.predictor.train()

    def eval(self):
        self.latent_tree.eval()
        self.predictor.eval() 

    def freeze(self, which='predictor'):

        if which == "skip":
            
            assert self.comp == concat_func, "Trying to freeze skip connection, but skip connection is not defined. Set <comp_func> to 'concatenate' instead."

            self.comp = freezed_concat_func

        else:

            if which == "predictor":
                block = self.predictor

            elif which == "latent_tree":
                block = self.latent_tree

            else:
                raise Exception("You can freeze either 'latent_tree' layer or 'predictor' layer.")

            for param in block.parameters():
                param.requires_grad = False

    def unfreeze(self, which='predictor'):

        if which == "skip":
            
            assert self.comp == freezed_concat_func or self.comp == concat_func, "Trying to unfreeze skip connection, but skip connection is not defined. Set <comp_func> to 'concatenate' instead."

            self.comp = concat_func

        else:

            if which == "predictor":
                block = self.predictor

            elif which == "latent_tree":
                block = self.latent_tree

            else:
                raise Exception("You can unfreeze either 'latent_tree' layer or 'predictor' layer.")

            for param in block.parameters():
                param.requires_grad = True

    def parameters(self):
        return list(self.latent_tree.parameters()) + list(self.predictor.parameters())

    def forward(self, X):

        z = self.latent_tree(X)[:, self.latent_tree.bst.leaves]

        xz = self.comp(X, z)

        return self.predictor(xz)

    def predict_bst(self, X):

        return self.latent_tree.predict(X)

    def db_distance(self, X):

        if self.latent_tree.split_func == 'linear':

            XA = self.latent_tree.split(X)
            return (XA / torch.norm(self.latent_tree.split[1].weight, dim=1, p=2))
        
        else:
            raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def load_model(cls, load_dir, add_load=None):
        
        checkpoint = torch.load(Path(load_dir) / 'model.t7', map_location=torch.device('cpu'))
        
        model = cls(**checkpoint, **checkpoint['kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])

        if add_load is not None:
            if 'optimizer' in add_load.keys():
                add_load['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

            if 'checkpoint' in add_load.keys():
                add_load['checkpoint'] = checkpoint

        model.latent_tree.bias_init = True 
        
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

        super(LTBinaryClassifier, self).__init__(bst_depth, in_size, pruned, **kwargs)

        self.params = locals()
        del self.params['self']

        # init predictor ( [x;z]-> y )
        self.predictor = LogisticRegression(self.pred_in_size, 1, linear, **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return y_pred.detach()

class LTClassifier(LTModel):

    def __init__(self, bst_depth, in_size, num_classes, pruned=True, linear=True, **kwargs):

        super(LTClassifier, self).__init__(bst_depth, in_size, pruned, **kwargs)

        self.params = locals()
        del self.params['self']

        # init predictor ( [x;z]-> y )
        if linear:
            self.predictor = Linear(self.pred_in_size, num_classes)
        else:
            self.predictor = MLP(self.pred_in_size, num_classes, **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred = torch.argmax(y_pred, axis=1)

        return y_pred.detach()

class LTRegressor(LTModel):

    def __init__(self, bst_depth, in_size, out_size, pruned=True, linear=True, **kwargs):

        super(LTRegressor, self).__init__(bst_depth, in_size, pruned, **kwargs)

        self.params = locals()
        del self.params['self']

        # init predictor ( [x;z]-> y )

        if linear:
            self.predictor = Linear(self.pred_in_size, np.prod(out_size))
        else:
            self.predictor = MLP(self.pred_in_size, np.prod(out_size), **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)

        return y_pred.detach()
