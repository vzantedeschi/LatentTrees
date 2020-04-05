import numpy as np
import cvxpy as cx

from scipy.special import softmax


def find_epsilons(X):
    Xs = np.sort(X, axis=0)
    Xd = np.diff(Xs, axis=0)
    Xd[Xd == 0] = np.inf
    return np.min(Xd, axis=0)


def qp(X, A, b, d_scores):
    descendent_l = [1, 3, 5]
    descendent_r = [2, 4, 6]
    ancestor_l = {
        0: [],
        1: [0],
        2: [],
        3: [0, 1],
        4: [0],
        5: [2],
        6: []
    }

    ancestor_r = {
        0: [],
        1: [],
        2: [0],
        3: [],
        4: [1],
        5: [0],
        6: [0, 2]
    }

    possible_split_nodes = [0, 1, 2]
    leaves = [3, 4, 5, 6]
    parent = [None, 0, 0, 1, 1, 2, 2]
    sibling = [None, 2, 1, 4, 3, 6, 5]

    eps = find_epsilons(X)
    eps_max = np.max(eps)

    n = X.shape[0]
    n_split = 3
    n_nodes = 7
    XA = X @ A.T  # n_samples by n_split
    epsA = eps @ A.T

    boolean = False
    regularize = False

    z = cx.Variable((n, n_nodes), boolean=boolean)
    l = cx.Variable(n_nodes, boolean=boolean)
    d = cx.Variable(n_nodes, boolean=boolean)

    N_min = 1

    constraints = []

    if not boolean:
        constraints.extend([
            z >= 0,
            z <= 1,
            l >= 0,
            l <= 1,
            d >= 0,
            d <= 1
        ])

    # tree compatibility
    for t in possible_split_nodes:
        for m in ancestor_r[t]:
            c = (XA[:, m] >= b[t] - 1 + z[:, t])
            constraints.append(c)

        for m in ancestor_l[t]:
            c = ((XA[:, m] + epsA[m]) <= b[t] + (1 + eps_max) * (1 - z[:, t]))
            constraints.append(c)

    # does node contain points
    for t in range(n_nodes):
        c = z[:, t] <= l[t]
        constraints.append(c)

    constraints.append(l <= d)
    constraints.append(cx.sum(z, axis=0) >= N_min * l)

    # structure of d and z
    for t in range(1, n_nodes):  # except root
        c = d[t] <= d[parent[t]]
        constraints.append(c)

        c = z[t, :] <= z[parent[t], :]
        constraints.append(c)

        c = z[t, :] + z[sibling[t], :] <= 1
        constraints.append(c)

    # print(constraints)
    obj = d @ d_scores
    if regularize:
        obj -= 0.5 * cx.sum_squares(z)
    pb = cx.Problem(cx.Maximize(obj), constraints)
    pb.solve(verbose=True)

    print(z.value)


def main():

    # 2d data, decision tree of depth D=2
    # max num nodes T = 2^(D+1) - 1 = 7 nodes.
    #
    #          0
    #         /  \
    #        1    2
    #       / \  / \
    #      3  4  5  6

    n = 10
    d = 2
    n_split = 3
    n_nodes = 7


    # given A, b, compute z (which leaf the data points fall into)
    X = np.random.rand(n, d)
    A = np.random.rand(n_split, d)
    b = np.random.rand(n_split)
    A = softmax(A, axis=1)

    is_split = [True] * 3 + [False] * 4
    descendent_l = [1, 3, 5]
    descendent_r = [2, 4, 6]

    def classify(x):
        k = 0

        while is_split[k]:
            val = np.dot(A[k], x) + b[k]
            k = descendent_l[k] if val < 0 else descendent_r[k]

        return k

    zz = [classify(x) for x in X]
    print(zz)
    # obj_val = np.sum(eta[np.arange(n), zz])
    # print(obj_val)

    d_scores = np.ones(n_nodes)

    qp(X, A, b, d_scores)


if __name__ == '__main__':
    main()
