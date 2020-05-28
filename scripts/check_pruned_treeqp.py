import numpy as np
import cvxpy as cx

def closed_form_1D(eta, qs):

    qs_gtr = qs[qs >= eta]
    ix = np.argsort(qs_gtr)[::-1]
    qs_srt = qs_gtr[ix]

    d = eta
    for k in range(len(qs_srt)):
        if d > qs_srt[k]:
            break
        d = (eta + np.sum(qs_srt[:k + 1])) / (k + 2)
    
    return d

def closed_form_colored(eta, qs):

    topk = 0
    nb_k = 0

    d = np.mean(eta)

    qs_srt = []
    for t in range(qs.shape[1]):

        qs_gtr = qs[qs[:, t] >= d, t]

        ix = np.argsort(qs_gtr)[::-1]
        qs_srt.append(qs_gtr[ix])

    qs_srt = sorted(np.hstack(qs_srt))[::-1]

    for k in range(len(qs_srt)):
        if d > qs_srt[k]:
            break
            
        topk += qs_srt[k]
        nb_k += 1

        d = (np.sum(eta) + np.sum(topk)) / (len(eta) + nb_k)
    
    return d

def closed_form(eta, qs, box=True, verbose=False):
    d = eta.copy()
    n = len(d)

    parent = [None, 0, 0, 1, 1, 2, 2]
    coloring = np.arange(n)

    for t, e in enumerate(eta):

        d[t] = closed_form_1D(e, qs[:, t]) 

    if verbose:
        print("init", d)

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
        d[pc_ix] = closed_form_colored(eta[pc_ix], qs[:, pc_ix])
        if verbose:
            print("joining", t, p, d)

    if box:
        d = np.clip(d, 0, 1)
    
    return d

def noq_closed_form(eta, box=True, verbose=False):

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

    if box:
        d = np.clip(d, 0, 1)
    
    return d

def solve_qp(eta, qs, box=True):
    parent = [None, 0, 0, 1, 1, 2, 2]
    N, T = qs.shape

    d = cx.Variable(eta.shape)
    z = cx.Variable(qs.shape)

    obj = .5 * cx.sum_squares(d - eta)
    obj += .5 * cx.sum_squares(z - qs)

    constr = [d[i] <= d[parent[i]] for i in range(1, T)]
    constr += [z[i] <= d for i in range(N)]

    if box:
        constr += [d >= 0, d <= 1, z >= 0, z <= 1]

    prob = cx.Problem(cx.Minimize(obj), constr)
    prob.solve()

    return d.value

def print_as_tree(d):
    print(d[0])
    print(d[1], "\t\t\t", d[2])
    print(d[3], d[4], d[5], d[6])

def main():
    SEED = 2020
    np.random.seed(SEED)

    etas = np.array([[1, -1, 0, 1, 2, -0.1, 0.3], [0, -1, 0, 1, 2, -0.1, 0.3], [0]*7, [1, 1, 0, 0, 0, 0, 0]])

    print("check qs = 0")
    qs = np.zeros((1, 7))

    for eta in etas:

        print("\neta", eta)
        print(closed_form(eta, qs, verbose=False))
        print(noq_closed_form(eta))
        print(solve_qp(eta, qs))

    qs = np.random.uniform(-1, 1, size = (10, 7))

    parent = [None, 0, 0, 1, 1, 2, 2]
    for t in range(1, 7):
        qs[:, t] = np.minimum(qs[:, t], qs[:, parent[t]])

    print("\ncheck with qs", qs)

    for eta in etas:

        print("\neta", eta)

        print(solve_qp(eta, qs))
        print(closed_form(eta, qs, verbose=True))

        print()

    from make_edge_cases import make_edge_cases

    etas = make_edge_cases(7)
    nb_cases = len(etas)
    print("checking {} edge cases...".format(nb_cases))

    for box in [True, False]:

        passed = 0
        if box:
            print("\nwith box constraints")
        else:
            print("\nwithout box constraints")

        for eta in etas:

            d_expected = solve_qp(eta, qs, box=box)
            d_obtained = closed_form(eta, qs, box=box, verbose=False)

            if not np.allclose(d_expected, d_obtained, atol=1e-3):
                print()
                print("eta", eta)
                print("qs", qs)
                d_obtained = closed_form(eta, qs, box=box, verbose=True)
                
                print(d_expected, np.sum((d_expected - eta) ** 2) + np.sum((qs - np.clip(qs, 0, d_expected)) ** 2))
                print(d_obtained, np.sum((d_obtained - eta) ** 2) + np.sum((qs - np.clip(qs, 0, d_obtained)) ** 2))

                print()
            else:
                passed += 1

        print("{} over {} cases passed".format(passed, nb_cases))

    NB_CASES = 1000
    passed = 0
    print("\nchecking {} random cases...".format(NB_CASES))
    for _ in range(NB_CASES):
        eta = np.random.uniform(-3, 3, size=7)
        qs = np.random.uniform(-1, 1, size = (10, 7))

        d_expected = solve_qp(eta, qs, box=True)
        d_obtained = closed_form(eta, qs, verbose=False)

        if not np.allclose(d_expected, d_obtained, atol=1e-4):
            print()
            print("eta", eta)
            print("qs", qs)
            d_obtained = closed_form(eta, qs, verbose=True)
            
            print(d_expected, np.sum((d_expected - eta) ** 2) + np.sum((qs - np.clip(qs, 0, d_expected)) ** 2))
            print(d_obtained, np.sum((d_obtained - eta) ** 2) + np.sum((qs - np.clip(qs, 0, d_obtained)) ** 2))

            print()
        else:
            passed += 1

    print("{} over {} cases passed".format(passed, NB_CASES))


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    main()

