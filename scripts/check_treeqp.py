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


# this is wrong, but the deep one handles this case correctly
def closed_form(eta):

    # tentative solution: d=-eta
    d = -eta.copy()

    # is it feasible? 3 cases
    if d[0] <= d[1] and d[0] <= d[2]:
        d[:] = (-eta).mean()

    elif d[0] <= d[1]:
        d[[0, 1]] = (-eta[[0, 1]]).mean()

    elif d[0] <= d[2]:
        d[[0, 2]] = (-eta[[0, 2]]).mean()

    return d


def main():

    # the network wants to weakly turn off the root and d[2]
    # but strongly wants to turn on d[1]
    # so in the solution, d[0] is turned.
    eta = np.array([.1, -10, .12], dtype=np.double)
    # eta = np.random.randn(3)
    # eta  /= np.linalg.norm(eta)

    print(eta)

    print("cvxpy")
    d = solve(eta, box=False)
    print(d)
    print(np.clip(d, a_min=0, a_max=1))
    print(solve(eta, box=True))

    print("closed form")
    d = closed_form(eta)
    print(d)
    print(np.clip(d, a_min=0, a_max=1))


def print_as_tree(d):
    print(d[0])
    print(d[1], "\t\t\t", d[2])
    print(d[3], d[4], d[5], d[6])


def deep_closed_form(eta, verbose=False):

    d = -eta
    n = len(d)
    parent = [None, 0, 0, 1, 1, 2, 2]
    coloring = np.arange(n)

    while True:
        max_violation = 0
        max_violation_ix = None

        for t in range(1, n):
            violation = max(d[t] - d[parent[t]], 0)
            if violation > max_violation:
                max_violation = violation
                max_violation_ix = t

        if max_violation == 0:
            # no more violations, we are done
            break

        # fix the worst violation
        t = max_violation_ix
        p = parent[t]
        pc = coloring[p]
        coloring[coloring == t] = pc
        d[coloring == pc] = np.mean(-eta[coloring == pc])
        if verbose:
            print("joining", t, p, d)

    return d


def deep_closed_form_(eta, verbose=False):
    coloring = np.arange(len(eta))
    parent = [None, 0, 0, 1, 1, 2, 2]
    sibling = [None, 2, 1, 4, 3, 6, 5]
    bottom_up = [3, 4, 5, 6, 1, 2, 0]

    d = -eta


# version that finalizes both children before tackling parent
# is incorrect
#
#    for i in bottom_up:
#         # update self and children, downward
#         val = np.mean(-eta[coloring == i])
#         d[coloring == i] = val
#         if verbose:
#             print(i, d)
#
#         # check for violation upward
#         p = parent[i]
#         if p is not None and d[i] > d[p]:
#             coloring[coloring == i] = p
#             if verbose:
#                 print('joining', i, 'into', p)

    children = [[1, 2], [3, 4], [5, 6], [], [], [], []]

    # greedy bottom-up, largest sibling first.
    for i in bottom_up:

        # swap with sibling
        if sibling[i] is not None and d[sibling[i]] > d[i]:
            i = sibling[i]
            bottom_up[bottom_up.index(sibling[i])] = i

        p = parent[i]
        if p is None:
            continue
        if d[i] > d[p]:
            coloring[coloring == i] = p
            val = np.mean(-eta[coloring == p])
            d[coloring == p] = val
            if verbose:
                print('joining', i, 'into', p, d)

    # top down propagate any possible created violations
    queue = [0]
    while len(queue):
        pp = queue.pop()
        mychildren = children[pp]

        for ii in mychildren:
            queue.append(ii)
            color = None
            if d[ii] > d[pp]:
                color = coloring[pp]
                coloring[coloring == ii] = color
                val = np.mean(-eta[coloring == color])
                d[coloring == color] = val
                if verbose:
                    print('td-join', pp, 'into', ii, d)


    return d


def wip():
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


def what():
    eta = np.array([0, 1, .25])

    for t in [.1, .2, .3, .4,  .5, .6, .7]:
        eta[1] = t
        print(solve(-eta, box=False), "(true)")
        # print(closed_form(-eta), "(incorrect)")
        eta2 = np.full(7, -99.0)
        eta2[:3] = eta
        print(deep_closed_form(-eta2, verbose=False)[:3])

    # eta = np.array([-.517, -.091, .303])
    # print(solve(-eta, box=False), "(true)")
    # eta2 = np.full(7, -99.0)
    # eta2[:3] = eta
    # print(deep_closed_form(-eta2, verbose=True))


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    # main()
    wip()
    # what()

