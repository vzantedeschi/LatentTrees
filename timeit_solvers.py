import numpy as np
import cvxpy as cx

import time
import torch
import tqdm

from src.qp import pruning_qp
from src.trees import BinarySearchTree

def make_edge_cases(d):

    edge_cases = []

    zero = np.zeros(d, dtype=np.double)
    bige = np.full(d, +100, dtype=np.double)
    smol = np.full(d, -100, dtype=np.double)

    for base in (zero, bige, smol):
        edge_cases.append(base)
        for i in range(d):
            v = base.copy()
            v[i] = 1
            edge_cases.append(v)

            v = base.copy()
            v[i] = 100
            edge_cases.append(v)

            v = base.copy()
            v[i] = -1
            edge_cases.append(v)

            v = base.copy()
            v[i] = -100
            edge_cases.append(v)

    return np.vstack(edge_cases)

def solve_qp(parents, eta, qs, box=True):

    N, T = qs.shape

    d = cx.Variable(eta.shape)
    z = cx.Variable(qs.shape)

    obj = .5 * cx.sum_squares(d - eta)
    obj += .5 * cx.sum_squares(z - qs)

    constr = [d[i] <= d[parents[i]] for i in range(1, T)]
    constr += [z[i] <= d for i in range(N)]

    if box:
        constr += [d >= 0, d <= 1, z >= 0, z <= 1]

    prob = cx.Problem(cx.Minimize(obj), constr)
    prob.solve()

    return d.value

if __name__ == "__main__":

    SEED = 2020
    np.random.seed(SEED)

    # fix n=100
    times_exact = np.zeros(7)
    times_cvx = np.zeros(7)

    for depth in tqdm.tqdm(range(7), desc="depth"):

        bst = BinarySearchTree(depth)
        parents = [bst.parent(t) for t in bst.nodes]

        etas = make_edge_cases(bst.nb_nodes)
        t_etas = [torch.from_numpy(eta).float() for eta in etas]

        num_points = 100
        qs = np.random.uniform(-1, 1, size=(num_points, bst.nb_nodes))

        t_qs = torch.from_numpy(qs).float()

        for eta, t_eta in zip(etas, t_etas):

            t0 = time.time()
            d_expected = solve_qp(parents, eta, qs)
            t1 = time.time()
            d_obtained = pruning_qp(t_qs, t_eta)
            t2 = time.time()

            times_exact[depth] += t2 - t1
            times_cvx[depth] += t1 - t0

    times_exact /= len(etas)
    times_cvx /= len(etas)

    np.save("cvx_times_n100.npy", times_cvx)
    np.save("exact_times_n100.npy", times_exact)

    # fix D=3
    times_exact = np.zeros(5)
    times_cvx = np.zeros(5)

    depth = 3
    bst = BinarySearchTree(depth)
    parents = [bst.parent(t) for t in bst.nodes]

    etas = make_edge_cases(bst.nb_nodes)
    t_etas = [torch.from_numpy(eta).float() for eta in etas]

    for n in tqdm.tqdm(range(5), desc="number of points"):

        num_points = 10**n
        qs = np.random.uniform(-1, 1, size=(num_points, bst.nb_nodes))
        t_qs = torch.from_numpy(qs).float()

        for eta, t_eta in zip(etas, t_etas):

            t0 = time.time()
            d_expected = solve_qp(parents, eta, qs)
            t1 = time.time()
            d_obtained = pruning_qp(t_qs, t_eta)
            t2 = time.time()

            times_exact[n] += t2 - t1
            times_cvx[n] += t1 - t0

    times_exact /= len(etas)
    times_cvx /= len(etas)

    np.save("cvx_times_D3.npy", times_cvx)
    np.save("exact_times_D3.npy", times_exact)