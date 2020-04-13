import numpy as np
import cvxpy as cx
import matplotlib.pyplot as plt

from scipy.special import softmax


def find_epsilons(X):
    Xs = np.sort(X, axis=0)
    Xd = np.diff(Xs, axis=0)
    Xd[Xd == 0] = np.inf
    return np.min(Xd, axis=0)

def plot_XOR(X, z, d, eta, boolean=False):
    fig = plt.figure()
    plt.scatter(X[:20,0], X[:20,1], s=50, c='red', label="y=0")
    plt.scatter(X[20:,0], X[20:,1], s=50, c='blue', label="y=1")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = fig.gca()
    ax.set_xticks(np.array([0,0.5,1.0]))
    ax.set_yticks(np.array([0,0.5,1.0]))
    plt.rc('grid', linestyle="--")
    plt.grid()
    zb = np.rint(z)
    db = np.rint(d)
    p = np.dot(zb, eta)
    pb= (p >= 0.5).astype(int)
    colors = np.array(['red', 'blue'])
    plt.scatter(X[:20,0], X[:20,1], s=20, c=list(colors[pb[:20].reshape(20,)]), edgecolors='black', label="p=0")
    plt.scatter(X[20:,0], X[20:,1], s=20, c=list(colors[pb[20:].reshape(20,)]), edgecolors='black', label="y=0")
    plt.savefig('result_bool=' + str(boolean) + '.png')


def qp(X, y, A, b, d_scores, alpha=1.0, boolean=False, regularize=False):
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

    #eps = find_epsilons(X)
    #eps_max = np.max(eps)
    mu = 0.005

    n = X.shape[0]
    n_split = 3
    n_nodes = 7
    XA = X @ A.T  # n_samples by n_split
    XAmu = (X + mu) @ A.T
    #epsA = eps @ A.T

    z = cx.Variable((n, n_nodes), boolean=boolean)
    d = cx.Variable(n_nodes, boolean=boolean)

    constraints = []

    if not boolean:
        constraints.extend([
            z >= 0,
            z <= 1,
            d >= 0,
            d <= 1,
        ])

    ## tree compatibility
    for t in range(n_nodes):#possible_split_nodes:
        for m in ancestor_r[t]:
            c = (XA[:, m] >= b[m] - 1 + z[:, t])# - (1-d[m]))
            constraints.append(c)

        for m in ancestor_l[t]:
            #c = ((XA[:, m] + epsA[m]) <= b[t] + (1 + eps_max) * (1 - z[:, t]))
            c = ((XAmu[:, m]) <= b[m] + (1 + mu) * (1 - z[:, t]))
            constraints.append(c)

    # is node active
    for t in range(n_nodes):#possible_split_nodes:
        c = z[:, t] <= d[t]
        constraints.append(c)

    # structure of d and z
    for t in range(1, n_nodes):  # except root
        c = d[t] <= d[parent[t]]
        constraints.append(c)
        c = z[:, t] <= z[:, parent[t]]
        constraints.append(c)
        c = z[:, t] + z[:, sibling[t]] <= 1
        constraints.append(c)

    c = z[:,0] == 1.0
    constraints.append(c)
    
    c = d[0] == 1.0
    constraints.append(c)

    ## objective (based on tree compatibility)
    obj = 0
    for t in range(n_nodes):#possible_split_nodes:
        for m in ancestor_r[t]:
            obj += cx.sum((XA[:, m] - b[m]) * z[:, t])

        for m in ancestor_l[t]:
            obj += cx.sum((b[m] - XA[:,m]) * z[:, t])
    
    obj = (1.0/(n_nodes*n))*obj

    if regularize:
        obj += (1.0/n_nodes)*(d @ d_scores)
    pb = cx.Problem(cx.Maximize(obj), constraints)
    pb.solve(verbose=True)
    print('qp res')
    print('z')
    print(np.round(z.value, 2))
    print('d')
    print(np.round(d.value, 2))
    if not boolean:
        print('z_round')
        print(np.rint(z.value))
        print('d_round')
        print(np.rint(d.value))
    return z.value, d.value


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
    # XOR
    np.random.seed(0)
    xp = 0.5*np.random.rand(n*2, 1) + 0.5
    xn = 0.5*np.random.rand(n*2, 1)
    yp = 0.5*np.random.rand(n*2, 1) + 0.5
    yn = 0.5*np.random.rand(n*2, 1)
    X = np.concatenate((xp[0:n], yp[0:n]), axis=1)
    X = np.concatenate((X, np.concatenate((xn[n:],  yn[n:]),  axis=1)))
    X = np.concatenate((X, np.concatenate((xp[n:],  yn[0:n]), axis=1)))
    X = np.concatenate((X, np.concatenate((xn[0:n], yp[n:]),  axis=1)))
    y = np.concatenate((np.zeros((20,1)), np.ones((20,1))), axis=0)

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
    print('zz')
    print(zz)

    d_scores = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0])#np.ones(n_nodes)

    z, d = qp(X, y, A, b, d_scores, boolean=True)

    print('does z respect the tree splits? all of the print statements below should be all 1s')
    print('----------------------------------------------------------------------------------')
    print('0->1')
    print(z[np.where((np.dot(X, A[0]) < b[0])==True)[0],1])
    print('0->2')
    print(z[np.where((np.dot(X, A[0]) >= b[0])==True)[0],2])
    ix1R = np.where((np.dot(X, A[0]) >= b[0])==True)[0]
    ix1L = np.where((np.dot(X, A[0]) <  b[0])==True)[0]
    print('1->4')
    print(z[ix1L[np.where((np.dot(X[ix1L], A[1]) >= b[1])==True)[0]],4])
    print('2->5')
    print(z[ix1R[np.where((np.dot(X[ix1R], A[2]) < b[2])==True)[0]],5])
    print('2->6')
    print(z[ix1R[np.where((np.dot(X[ix1R], A[2]) >= b[2])==True)[0]],6])
    print('the following should be all 0s')
    print('------------------------------')
    print('1->3')
    print(z[ix1L[np.where((np.dot(X[ix1L], A[1]) >= b[1])==True)[0]],3])



if __name__ == '__main__':
    main()
