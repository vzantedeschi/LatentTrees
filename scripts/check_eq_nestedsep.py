import numpy as np
import matplotlib.pyplot as plt

def solve(eta, qs):
    qs_gtr = qs[qs >= eta]
    ix = np.argsort(qs_gtr)[::-1]
    qs_srt = qs_gtr[ix]

    d = eta
    for k in range(len(qs_srt)):
        if d > qs_srt[k]:
            break
        topk = qs_srt[:k + 1]
        d = (eta + np.sum(topk)) / (k + 2)
    return d


def main():
    eta = .5

    ds = np.linspace(-1, 2, 5000)
    def curve(qs):
        def f(d):
            val = (d - eta) ** 2
            val += np.sum((d - qs[qs >= d]) ** 2)
            return val / 2
        curve = [f(d) for d in ds]
        print("numeric min", ds[np.argmin(curve)])
        print("analytic min", solve(eta, qs))
        print()
        return curve

    qs = np.array([.4])
    plt.plot(ds, curve(qs), label=str(qs))

    qs = np.array([1.0])
    plt.plot(ds, curve(qs), label=str(qs))

    qs = np.array([.4, 1.0])
    plt.plot(ds, curve(qs), label=str(qs))

    qs = np.array([3.0, 1.0])
    plt.plot(ds, curve(qs), label=str(qs))

    qs = np.array([2.0, 1.0])
    plt.plot(ds, curve(qs), label=str(qs))

    qs = np.array([2.0, 1.2, 1.2, 1])
    plt.plot(ds, curve(qs), label=str(qs))

    qs = np.array([2.0, 1.5, 1.2, 1])
    plt.plot(ds, curve(qs), label=str(qs))

    qs = np.array([2.0, 2.0])
    plt.plot(ds, curve(qs), label=str(qs))

    plt.plot(ds, 0.5 * (ds - eta) ** 2, ls=":")
    plt.legend()
    plt.show()





if __name__ == '__main__':
    main()

