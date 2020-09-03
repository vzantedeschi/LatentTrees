import numpy as np

from sklearn.datasets import make_swiss_roll

# ------------------------------------------------------------ TOY DATASETS

def toy_dataset(n=1000, distr="xor", dim=2):

    if distr == "xor":
        
        X = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n, dim))
        Y = (X[:,0] * X[:,1] >= 0).astype(int)

        return dict(X=X.astype(np.float32), Y=Y)

    elif distr == "reg-xor":
        
        X = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n, dim))
        labels = (X[:,0] * X[:,1] >= 0)

        Y = np.empty(n)
        Y[labels] = np.random.normal(0.8, 0.1, np.sum(labels))
        Y[~labels] = np.random.normal(0.2, 0.1, np.sum(~labels))

        return dict(X=X.astype(np.float32), Y=Y.astype(np.float32), labels=labels)

    elif distr == "swissroll":

        n2 = n // 2

        X1,_ = make_swiss_roll(n_samples=n2, noise=0)
        Y1 = np.ones(n2)

        X2 = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n2, dim))
        Y2 = np.zeros(n2)

        X = np.r_[X1[:,::2] / 15, X2]
        Y = np.r_[Y1, Y2]

        return dict(X=X.astype(np.float32), Y=Y)

    else:
        NotImplementedError