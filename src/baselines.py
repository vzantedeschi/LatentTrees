import numpy as np
import pandas as pd

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