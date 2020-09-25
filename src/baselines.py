import numpy as np
import pandas as pd
import torch

from functools import reduce

from pyoptree.optree import OptimalTreeModel

from src.trees import BinarySearchTree

class OptTree():

    def __init__(self, bst_depth, dim, N_min=1, verbose=False):

        self.in_size = dim
        self.in_columns = [f"x{i}" for i in range(dim)]
        self.out_columns = ["y"]

        self.bst = BinarySearchTree(bst_depth)      
        self.optree = OptimalTreeModel(self.in_columns, self.out_columns[0], tree_depth=bst_depth, N_min=N_min)
        self.verbose = verbose

    def train(self, x, y):

        dataframe = pd.DataFrame(data=np.hstack((x, y[:, None])), columns=self.in_columns + self.out_columns)
        self.optree.train(dataframe, train_method="mio", show_training_process=self.verbose)

        self.A = np.zeros((self.bst.nb_split, self.in_size + 1))

        for t in self.bst.split_nodes:
            self.A[t, :-1] = np.array(self.optree.a[t+1])
            self.A[t, -1] = -self.optree.b[t+1]

    def predict(self, x):

        dataframe = pd.DataFrame(data=x, columns=self.in_columns)

        y_pred = self.optree.predict(dataframe).loc[:, 'prediction'].values
        
        return y_pred

    def predict_bst(self, x):

        x = np.hstack((x, np.ones((len(x), 1))))

        z = self._compute_z(x)

        return z, self.bst.predict(z)

    def _compute_z(self, x):

        # compute tree paths q
        XA = x.dot(self.A.T)

        z = np.ones((len(x), self.bst.nb_nodes))

        z[:, self.bst.desc_left] = (XA[:, self.bst.split_nodes] < 0).astype(int)
        z[:, self.bst.desc_right] = (XA[:, self.bst.split_nodes] >= 0).astype(int)

        # upper bound children's q to parent's q
        for _ in range(self.bst.depth):
            z[:, self.bst.desc_left] = np.minimum(z[:, self.bst.desc_left], z[:, self.bst.split_nodes])
            z[:, self.bst.desc_right] = np.minimum(z[:, self.bst.desc_right], z[:, self.bst.split_nodes])

        return z

""" Code adapted from https://github.com/wOOL/DNDT/blob/master/pytorch/demo.ipynb"""

def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res

def torch_bin(x, cut_points, temperature=0.1):
    # x is a N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = cut_points.shape[0]
    W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points, _ = torch.sort(cut_points)  # make sure cut_points is monotonically increasing
    b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0),0)
    h = torch.matmul(x, W) + b
    res = torch.exp(h-torch.max(h))
    res = res/torch.sum(res, dim=-1, keepdim=True)
    return h

class DNDT(torch.nn.Module):

    def __init__(self, in_size, num_classes, num_cuts=1, temperature=0.1):

        super().__init__()

        self.params = locals()
        del self.params['self']

        self.cuts = [num_cuts] * in_size
        self.num_leaves = np.prod(np.array(self.cuts) + 1)
        self.num_classes = num_classes
        self.temperature = temperature

        self.cut_points_list = [torch.nn.Parameter(torch.rand([i], requires_grad=True)) for i in self.cuts]
        self.leaf_score = torch.nn.Parameter(torch.rand([self.num_leaves, num_classes], requires_grad=True))

    def forward(self, x):

        leaf = reduce(torch_kron_prod,
                  map(lambda z: torch_bin(x[:, z[0]:z[0] + 1], z[1], self.temperature), enumerate(self.cut_points_list)))
        
        return torch.matmul(leaf, self.leaf_score)

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