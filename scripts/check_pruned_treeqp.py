import numpy as np
import cvxpy as cx

def closed_form_1D(eta, qs):

    qs_gtr = qs[qs >= eta]
    ix = np.argsort(qs_gtr)[::-1]
    qs_srt = qs_gtr[ix]

    d = eta
    
    k = 0
    for k in range(len(qs_srt)):
        if d > qs_srt[k]:
            k += 1
            break

    topk = qs_srt[:k]
    d = (eta + np.sum(topk)) / (k + 1)
    
    return d

def closed_form_colored(d, eta, qs):

    topk = 0
    nb_k = 0
    for d_t, q_t in zip():

        qs_gtr = q_t[q_t >= d_t]
        ix = np.argsort(qs_gtr)[::-1]
        qs_srt = qs_gtr[ix]
        
        for k in range(len(qs_srt)):
            if d_t > qs_srt[k]:
                break
                
            topk += qs_srt[k]
            nb_k += 1

    new_d = (np.sum(eta) + np.sum(topk)) / (len(eta) + nb_k)
    
    return new_d

def closed_form(eta, qs, verbose=False):

    d = eta.copy()
    n = len(d)

    parent = [None, 0, 0, 1, 1, 2, 2]
    coloring = np.arange(n)

    for t, e in enumerate(eta):

        d[t] = closed_form_1D(e, qs[:, t]) 

    while True:
        max_violating_d = -np.inf
        max_violating_ix = None

        for t in range(1, n):
            # if edge is violating, and is larger than max so far
            if d[t] > d[parent[t]] and d[t] > max_violating_d:
                max_violating_d = d[t]
                max_violating_ix = t

        if max_violating_ix is None:
            # no more violations, we are done
            break

        # fix the selected violating edge, propagating along color.
        # invariant: always keep the color of the parent.
        t = max_violating_ix
        p = parent[t]
        pc = coloring[p]
        coloring[coloring == t] = pc

        pc_ix = (coloring == pc)
        d[pc_ix] = closed_form_colored(d[pc_ix], eta[pc_ix], qs[:, pc_ix])
        if verbose:
            print("joining", t, p, d)

    return np.clip(d, 0, 1)

def noq_closed_form(eta, verbose=False):

    d = eta.copy()
    n = len(d)

    # todo: generalize parent vector by depth. Rest of code is ok
    # IIRC from college this can be done with some modulo arithmetic.
    parent = [None, 0, 0, 1, 1, 2, 2]
    coloring = np.arange(n)

    while True:
        max_violating_d = -np.inf
        max_violating_ix = None

        for t in range(1, n):
            # if edge is violating, and is larger than max so far
            if d[t] > d[parent[t]] and d[t] > max_violating_d:
                max_violating_d = d[t]
                max_violating_ix = t

        if max_violating_ix is None:
            # no more violations, we are done
            break

        # fix the selected violating edge, propagating along color.
        # invariant: always keep the color of the parent.
        t = max_violating_ix
        p = parent[t]
        pc = coloring[p]
        coloring[coloring == t] = pc
        d[coloring == pc] = np.mean(eta[coloring == pc])
        if verbose:
            print("joining", t, p, d)

    return np.clip(d, 0, 1)

# def solve_qp(eta, qs, box=False):
#     parent = [None, 0, 0, 1, 1, 2, 2]
#     N, T = qs.shape

#     d = cx.Variable(eta.shape)

#     obj = .5 * cx.sum_squares(d - eta)

#     violations = cx.transforms.indicator([qs[i] >= d for i in range(N)])
#     obj += .5 * cx.sum_squares([(d - qs[i])**2 for i in range(N)])

#     constr = [d[i] <= d[parent[i]] for i in range(1, T)]

#     prob = cx.Problem(cx.Minimize(obj), constr)
#     prob.solve()
#     return d.value

def print_as_tree(d):
    print(d[0])
    print(d[1], "\t\t\t", d[2])
    print(d[3], d[4], d[5], d[6])

def main():
    SEED = 2020
    np.random.seed(SEED)

    qs = np.random.uniform(0, 1, size = (10, 7))
    print("qs", qs)

    etas = np.array([[1, -1, 0, 1, 2, -0.1, 0.3], [0, -1, 0, 1, 2, -0.1, 0.3], [0]*7, [1, 1, 0, 0, 0, 0, 0]])

    for eta in etas:

        print("\neta", eta)

        closed_form(eta, qs, verbose=True)
        # solve_qp(eta, qs)

        print()

    print("\ncheck with no qs")
    qs = np.zeros((1, 7))

    for eta in etas:

        print("\neta", eta)
        print(closed_form(eta, eta[None,], verbose=False))
        print(noq_closed_form(eta))

    for _ in range(10000):
        eta = np.random.uniform(-3, 3, size = (7))

        d_expected = noq_closed_form(eta)
        d_obtained = closed_form(eta, qs)

        if not np.allclose(d_expected, d_obtained):
            print()
            print("eta", eta)
            d_obtained = closed_form(eta, eta[None,], verbose=True)
            print(d_expected, np.sum((d_expected - eta) ** 2))
            print(d_obtained, np.sum((d_obtained - eta) ** 2))
            # print_as_tree(d_obtained)
            print()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    main()

