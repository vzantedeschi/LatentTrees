import numpy as np
import cvxpy as cx

import time
import torch
import tqdm

from cvxpylayers.torch import CvxpyLayer

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
    obj += .5 * cx.sum_squares(z - qs - 0.5)

    constr = [d[i] <= d[parents[i]] for i in range(1, T)]
    constr += [z[i] <= d for i in range(N)]

    if box:
        constr += [d >= 0, d <= 1, z >= 0, z <= 1]

    prob = cx.Problem(cx.Minimize(obj), constr)
    prob.solve()

    return d.value

def solve_qpth(parents, eta, qs, box=False):

    N, T = qs.shape

    a = cx.Variable(eta.shape)
    z = cx.Variable(qs.shape)

    e = cx.Parameter(shape=eta.shape)
    q = cx.Parameter(shape=qs.shape)

    obj = .5 * cx.sum_squares(a - e)
    obj += .5 * cx.sum_squares(z - q - 0.5)

    constr = [a[i] <= a[parents[i]] for i in range(1, T)]
    constr += [z[i] <= a for i in range(N)]

    if box:
        constr += [a >= 0, a <= 1, z >= 0, z <= 1]

    problem = cx.Problem(cx.Minimize(obj), constr)

    cvxpylayer = CvxpyLayer(problem, parameters=[e, q], variables=[a, z])

    # solve the problem
    a, z = cvxpylayer(eta, qs)

    return a

if __name__ == "__main__":

    SEED = 2020
    np.random.seed(SEED)

    # n fixed to 100
    num_points = 100

    times_exact = np.zeros(7)
    times_cvx = np.zeros(7)

    for depth in tqdm.tqdm(range(7), desc="depth"):

        bst = BinarySearchTree(depth)
        parents = [bst.parent(t) for t in bst.nodes]

        etas = make_edge_cases(bst.nb_nodes)
        t_etas = [torch.tensor(eta, requires_grad=True).float() for eta in etas]

        qs = np.random.uniform(-1, 1, size=(num_points, bst.nb_nodes))

        t_qs = torch.tensor(qs, requires_grad=True).float()

        # for t in range(1, 7):
        #     qs[:, t] = np.minimum(qs[:, t], qs[:, parents[t]])

        for eta, t_eta in zip(etas, t_etas):

            a_ours = pruning_qp(t_qs, t_eta)
            a_cvx = solve_qpth(parents, t_eta, t_qs)

            # timing backward pass
            t0 = time.time()
            a_ours.sum().backward()
            t1 = time.time()
            a_cvx.sum().backward()
            t2 = time.time()

            times_exact[depth] += t2 - t1
            times_cvx[depth] += t1 - t0

        times_exact[depth] /= len(etas)
        times_cvx[depth] /= len(etas)

        np.save("n100_cvx_times.npy", times_cvx)
        np.save("n100_exact_times.npy", times_exact)

    # D fixed to 3
    depth = 3

    times_exact = np.zeros(4)
    times_cvx = np.zeros(4)
    
    for n in tqdm.tqdm(range(4), desc="number of points"):

        num_points = 10**n

        bst = BinarySearchTree(depth)
        parents = [bst.parent(t) for t in bst.nodes]

        etas = make_edge_cases(bst.nb_nodes)
        t_etas = [torch.tensor(eta, requires_grad=True).float() for eta in etas]

        qs = np.random.uniform(-1, 1, size=(num_points, bst.nb_nodes))

        t_qs = torch.tensor(qs, requires_grad=True).float()

        for eta, t_eta in zip(etas, t_etas):

            a_ours = pruning_qp(t_qs, t_eta)
            a_cvx = solve_qpth(parents, t_eta, t_qs)

            # timing backward pass
            t0 = time.time()
            a_ours.sum().backward()
            t1 = time.time()
            a_cvx.sum().backward()
            t2 = time.time()

            times_exact[n] += t2 - t1
            times_cvx[n] += t1 - t0

        times_exact[n] /= len(etas)
        times_cvx[n] /= len(etas)

        np.save("D3_cvx_times.npy", times_cvx)
        np.save("D3_exact_times.npy", times_exact)
