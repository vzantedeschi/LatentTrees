import torch
from torch.autograd.functional import vjp

from .trees import BinarySearchTree
from .qp import pruning_qp, pruning_qp_slow


def make_data(depth, n_samples=13, seed=42):
    n_nodes = 2 ** (depth + 1) - 1
    torch.manual_seed(seed)
    # distances from node boundaries
    q = torch.randn(n_samples, n_nodes,
                    dtype=torch.float32,
                    requires_grad=True)

    # pruning strengths
    eta = torch.randn(n_nodes,
                      dtype=torch.float32,
                      requires_grad=True)

    return q, eta


def test_pruning_qp_smoke():
    depth = 3
    q, eta = make_data(depth)
    d = pruning_qp(q, eta)


def test_pruning_qp_match_slow_fwd():
    """Test c++ vs pure python algo: forward pass"""
    depth = 3

    for seed in range(100):
        q, eta = make_data(depth, seed=seed)
        d = pruning_qp(q, eta)
        d_slow = pruning_qp_slow(q, eta, BinarySearchTree(depth))
        assert torch.allclose(d, d_slow)


def test_pruning_qp_match_slow_backward():
    """Test c++ vs pure python algo: backward pass"""
    depth = 3
    n_nodes = 2 ** (depth + 1) - 1

    torch.manual_seed(42)
    grad_ds = torch.randn(23, n_nodes)
    grad_ds /= torch.norm(grad_ds, dim=1).unsqueeze(1)

    def pruning_qp_sl_(q, eta):
        return pruning_qp_slow(q, eta, BinarySearchTree(depth))

    for seed in range(10):
        data = make_data(depth, seed=seed)

        for k in range(grad_ds.shape[0]):
            grad_d = grad_ds[k]
            _, (grad_q_fa, grad_eta_fa) = vjp(pruning_qp, data, grad_d)
            _, (grad_q_sl, grad_eta_sl) = vjp(pruning_qp_sl_, data, grad_d)
            assert torch.allclose(grad_q_fa, grad_q_sl)
            assert torch.allclose(grad_eta_fa, grad_eta_sl)

