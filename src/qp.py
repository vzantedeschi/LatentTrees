import numpy as np

import torch

from .trees import BinarySearchTree
from .qp_fast import compute_d_fast


class LatentDT(torch.nn.Module):

    def __init__(self, bst_depth, dim, pruned=True):

        super().__init__()

        self.bst = BinarySearchTree(bst_depth)
        self.A = torch.nn.Parameter(torch.rand(self.bst.nb_split, dim))

        self.pruned = pruned
        if pruned:
            self.eta = torch.nn.Parameter(torch.rand(self.bst.nb_nodes))

    def forward(self, x):

        q = self._compute_q(x)

        if self.pruned:

            self.d = d_slow = self._compute_d_slow(q)
            print("d slow", self.d)
            self.d = d_fast = self._compute_d_fast(q)
            print("d fast", self.d)
            exit()
            print(torch.norm(d_slow - d_fast).item())
            exit()

            z = torch.clamp(q, 0, 1)
            z = torch.min(z, self.d)

        else:
            z = torch.clamp(q, 0, 1)

        return z

    def predict(self, x):

        z = self.forward(x).detach().numpy()

        return self.bst.predict(z)

    def _compute_d_slow(self, q):

        # init d and colors different for all nodes
        d = self.eta.clone()
        coloring = np.arange(self.bst.nb_nodes)

        for c in coloring:
            d[c] = self._compute_d_colored_slow(q, [c])

        n_iter = 0

        while True:
            n_iter += 1
            max_violating_d = - np.inf
            max_violating_ix = None

            for t in range(1, self.bst.nb_nodes):
                # if edge is violating, and is larger than max so far
                p = self.bst.parent(t)
                if d[t] > d[p] and d[t] > max_violating_d:
                    max_violating_d = d[t]
                    max_violating_ix = t

            if max_violating_ix is None:
                # no more violations, we are done
                break

            # fix the selected violating edge, propagating along color.
            # invariant: always keep the color of the parent.
            p = self.bst.parent(max_violating_ix)
            pc = coloring[p]
            coloring[coloring == max_violating_ix] = pc
            print("join ", max_violating_ix, "into", p)
            # print(coloring)

            pc_ix = (coloring == pc)
            d[pc_ix] = self._compute_d_colored_slow(q, pc_ix)

        d = torch.clamp(d, 0, 1)
        print("Took", n_iter, "iter")

        return d

    def _compute_d_colored_slow(self, q, idx):

        topk = 0
        nb_k = 0

        d = torch.mean(self.eta[idx])

        # select qs greater than current d (violating the constraints)
        q_sorted, _ = torch.sort(q[:, idx][q[:, idx] >= d], descending=True)

        for k in range(len(q_sorted)):
            if d > q_sorted[k]:
                break
            topk += q_sorted[k]
            nb_k += 1

            d = (torch.sum(self.eta[idx]) + topk) / (len(self.eta[idx]) + nb_k)

        return d

    def _compute_d_fast(self, q):
        d = compute_d_fast(q, self.eta)
        return d

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

def main():
    # torch.manual_seed(42)
    batch_size = 5
    dim = 3

    X = torch.randn(batch_size, dim)

    dt = LatentDT(bst_depth=3, dim=dim)
    Z = dt(X)


if __name__ == '__main__':
    main()


