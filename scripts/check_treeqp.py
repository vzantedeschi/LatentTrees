import numpy as np
import cvxpy as cx


def solve(eta, box=False):
    d = cx.Variable(len(eta))
    obj = .5 * cx.sum_squares(d + eta)
    constr = [d[1] <= d[0], d[2] <= d[0]]

    if box:
        constr.append(d >= 0)
        constr.append(d <= 1)

    prob = cx.Problem(cx.Minimize(obj), constr)
    prob.solve()
    return d.value.round(3)


def solve2(eta, box=False):
    parent = [None, 0, 0, 1, 1, 2, 2]
    n = len(eta)
    d = cx.Variable(n)
    obj = .5 * cx.sum_squares(d + eta)
    constr = [d[i] <= d[parent[i]] for i in range(1, n)]

    if box:
        constr.append(d >= 0)
        constr.append(d <= 1)

    prob = cx.Problem(cx.Minimize(obj), constr)
    prob.solve()
    return d.value


def print_as_tree(d):
    print(d[0])
    print(d[1], "\t\t\t", d[2])
    print(d[3], d[4], d[5], d[6])


def deep_closed_form(eta, verbose=False):

    d = -eta
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
        d[coloring == pc] = np.mean(-eta[coloring == pc])
        if verbose:
            print("joining", t, p, d)

    return d


def main():
    eta = np.array([0, -1, 0, .1, -.1, 0, -3], dtype=np.double)
    print(solve2(eta, box=False))
    deep_closed_form(eta, verbose=True)

    print()
    print()

    eta = np.array([0, -1, 0, .1, -.1, 0, -0.5], dtype=np.double)
    print(solve2(eta, box=False))
    deep_closed_form(eta, verbose=True)

    print()
    print()

    rng = np.random.RandomState(42)
    for _ in range(1000):
        eta = rng.randn(7)
        eta /= np.sqrt(np.sum(eta ** 2))

        d_expected = solve2(eta, box=False)
        d_obtained = deep_closed_form(eta, verbose=False)

        if not np.allclose(d_expected, d_obtained):
            print()
            print(-eta)
            d_obtained = deep_closed_form(eta, verbose=True)
            print(d_expected, np.sum((d_expected + eta) ** 2))
            print(d_obtained, np.sum((d_obtained + eta) ** 2))
            # print_as_tree(d_obtained)
            print()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    main()

