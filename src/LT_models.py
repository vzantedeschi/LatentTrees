import numpy as np
import torch

from pathlib import Path

from src.trees import BinarySearchTree
from src.monitors import MonitorTree
from src.qp import pruning_qp
from src.utils import *

# ----------------------------------------------------------------------- LATENT TREE LAYER

class LatentTree(torch.nn.Module):

    def __init__(self, bst_depth, dim, reg=0, **split_args):

        super(LatentTree, self).__init__()

        self.bst = BinarySearchTree(bst_depth)      

        self.bias = torch.nn.Parameter(torch.zeros(self.bst.nb_split), requires_grad=True)
        self.bias_init = False
        
        self.act_type = split_args.pop("split_act", "none")
        self.split_type = split_args.pop("split_func", "linear")

        self.act = act_dict[self.act_type]
        self.split = split_dict[self.split_type](dim, self.bst.nb_split, **split_args)
        
        self.eta = reg

    def forward(self, x):

        if not self.bias_init:
            self._init_bias(x)
            self.bias_init = True

        q = self._compute_q(x)
        z = torch.clamp(q, 0, 1)

        if self.eta > 0:
            # import pdb; pdb.set_trace()
            self.d = pruning_qp(q.cpu(), self.eta).to(x.device) # pruning runs only on cpu
            self.d = torch.clamp(self.d, 0, 1) # apply box constraints
            z = torch.min(z, self.d) # prune tree traversals

        self.z = z
        self.q = q

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

            s_p = (s[:, p] + bias[p]).cpu().detach().numpy() # parent's projections

            node_masks.append(node_masks[p] & (s_p > 0)) # points going to the left node
            node_masks.append(node_masks[p] & (s_p < 0)) # points going to the rigth node

            if sum(node_masks[l]) > 0:
                bias[l] = -s[node_masks[l], l].mean()

            if sum(node_masks[r]) > 0:
                bias[r] = -s[node_masks[r], r].mean()

        self.bias = torch.nn.parameter.Parameter(bias, requires_grad=True)

# ------------------------------------------------------------------------------- NN MODELS

class LTModel(torch.nn.Module):

    def __init__(self, bst_depth, in_size, reg, **kwargs):

        torch.nn.Module.__init__(self)

        comp_func = kwargs.pop('comp_func', 'concatenate')

        # init latent tree optimizer (x -> z)
        self.latent_tree = LatentTree(bst_depth, in_size, reg, **kwargs)

        # init composition function for constructing input for predictor
        if comp_func != "concatenate":

            if comp_func == "none":
                self.comp = none_func
                self.pred_in_size = self.latent_tree.bst.nb_nodes # predictor's input size

            else:
                raise NotImplementedError

        else:
            self.comp = concat_func
            self.pred_in_size = np.prod(in_size) + self.latent_tree.bst.nb_nodes # predictor's input size

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

        z = self.latent_tree(X)

        xz = self.comp(X, z)

        return self.predictor(xz)

    def predict_bst(self, X):

        return self.latent_tree.predict(X)

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

    def __init__(self, bst_depth, in_size, reg=0, linear=True, **kwargs):

        super(LTBinaryClassifier, self).__init__(bst_depth, in_size, reg, **kwargs)

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

    def __init__(self, bst_depth, in_size, num_classes, reg=0, linear=True, **kwargs):

        super(LTClassifier, self).__init__(bst_depth, in_size, reg, **kwargs)

        self.params = locals()
        del self.params['self']

        # init predictor ( [x;z]-> y )
        if linear:
            self.predictor = torch.nn.Linear(self.pred_in_size, num_classes)
        else:
            self.predictor = MLP(self.pred_in_size, num_classes, **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)
        y_pred = torch.argmax(y_pred, axis=1)

        return y_pred.detach()

class LTRegressor(LTModel):

    def __init__(self, bst_depth, in_size, out_size, reg=0, linear=True, **kwargs):

        super(LTRegressor, self).__init__(bst_depth, in_size, reg, **kwargs)

        self.params = locals()
        del self.params['self']

        # init predictor ( [x;z]-> y )

        if linear:
            self.predictor = torch.nn.Linear(self.pred_in_size, np.prod(out_size))
        else:
            self.predictor = MLP(self.pred_in_size, np.prod(out_size), **kwargs)

    def predict(self, X):

        y_pred = self.forward(X)

        return y_pred.detach()
