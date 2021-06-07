import numpy as np

import torch

from .trees import BinarySearchTree
from .qp_fast import compute_d_fast


class PruningQPFast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, eta):
        d, ctx.state = compute_d_fast(q, eta)
        ctx.save_for_backward(q)
        return d

    @staticmethod
    def backward(ctx, grad_d):

        colors = ctx.state.color
        denoms = ctx.state.denoms
        indices = ctx.state.color_to_ix

        # we consider the mapping as the composition of two functions f and g
        # f(q, eta) -> d_colors , with shape (n_unique_colors)
        # g(d_colors) -> d which copies each color to all joined indices.

        # we first apply the Jacobian of g(d_colors)
        # by accummulating grad_d for each color.
        colors = np.array(colors)
        color_uniq, color_inv = np.unique(colors, return_inverse=True)
        n_colors = len(color_uniq)
        grad_d_colors = torch.zeros(n_colors)
        grad_d_colors.index_add_(0, torch.LongTensor(color_inv), grad_d)

        # next,
        # partial d_colors[c] / q[i,k] = 1/denom[c] if (i,k) in indices[c]
        q, = ctx.saved_tensors
        # grad_eta = torch.zeros_like(eta)
        grad_q = torch.zeros_like(q)

        for c in range(n_colors):
            color = color_uniq[c]  # map back to discontinuous index

            rows = [k for _, k in indices[color]]
            cols = [i for i, _ in indices[color]]
            grad_q[rows, cols] += grad_d_colors[c] / denoms[color]

        return grad_q, None


def pruning_qp(q, eta):
    """Pruning QP: fast cpp impl"""
    return PruningQPFast.apply(q, eta)


def _pruning_qp_subproblem(q, eta, idx):
    topk = 0
    nb_k = 0

    d = torch.mean(eta[idx])

    # select qs greater than current d (violating the constraints)
    q_sorted, _ = torch.sort(q[:, idx][q[:, idx] >= d], descending=True)

    for k in range(len(q_sorted)):
        if d > q_sorted[k]:
            break
        topk += q_sorted[k]
        nb_k += 1

        d = (torch.sum(eta[idx]) + topk) / (len(eta[idx]) + nb_k)

    return d


def pruning_qp_slow(q, eta, bst):
    # init d and colors different for all nodes
    d = eta.clone()
    coloring = np.arange(bst.nb_nodes)

    for c in coloring:
        d[c] = _pruning_qp_subproblem(q, eta, [c])

    n_iter = 0

    while True:
        n_iter += 1
        max_violating_d = -np.inf
        max_violating_ix = None

        for t in range(1, bst.nb_nodes):
            # if edge is violating, and is larger than max so far
            p = bst.parent(t)
            if d[t] > d[p] and d[t] > max_violating_d:
                max_violating_d = d[t]
                max_violating_ix = t

        if max_violating_ix is None:
            # no more violations, we are done
            break

        # fix the selected violating edge, propagating along color.
        # invariant: always keep the color of the parent.
        p = bst.parent(max_violating_ix)
        pc = coloring[p]
        coloring[coloring == max_violating_ix] = pc

        pc_ix = (coloring == pc)
        d[pc_ix] = _pruning_qp_subproblem(q, eta, pc_ix)

    return d


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

            self.d = pruning_qp(q, self.eta)
            self.d = torch.clamp(self.d, 0, 1)
            z = torch.clamp(q, 0, 1)
            z = torch.min(z, self.d)

        else:
            z = torch.clamp(q, 0, 1)

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
        q[:, self.bst.desc_left] = torch.min(q[:, self.bst.split_nodes],
                                             XA[:, self.bst.split_nodes])
        q[:, self.bst.desc_right] = torch.min(q[:, self.bst.split_nodes],
                                              -XA[:, self.bst.split_nodes])

        for _ in range(self.bst.depth):
            q[:, self.bst.desc_left] = torch.min(q[:, self.bst.desc_left],
                                                 q[:, self.bst.split_nodes])
            q[:, self.bst.desc_right] = torch.min(q[:, self.bst.desc_right],
                                                  q[:, self.bst.split_nodes])

        return q


def main():
    # torch.manual_seed(42)
    batch_size = 5
    dim = 3

    X = torch.randn(batch_size, dim)

    dt = LatentDT(bst_depth=3, dim=dim)
    Z = dt(X)
    print(Z)


if __name__ == '__main__':
    main()


