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


# not fully worked out
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


if __name__ == '__main__':
    main()

