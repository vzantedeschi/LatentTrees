import numpy as np

from pathlib import Path

from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split

from src.utils import download

# ------------------------------------------------------------ TOY DATASETS

def toy_dataset(n=1000, distr="xor", dim=2):

    if distr == "xor":
        
        X = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n, dim))
        Y = (X[:,0] * X[:,1] >= 0).astype(int)

        return X, Y

    elif distr == "reg-xor":
        
        X = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n, dim))
        labels = (X[:,0] * X[:,1] >= 0)

        Y = np.empty(n)
        Y[labels] = np.random.normal(0.8, 0.1, np.sum(labels))
        Y[~labels] = np.random.normal(0.2, 0.1, np.sum(~labels))

        return X, Y, labels

    elif distr == "swissroll":

        n2 = n // 2

        X1,_ = make_swiss_roll(n_samples=n2, noise=0)
        Y1 = np.ones(n2)

        X2 = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n2, dim))
        Y2 = np.zeros(n2)

        X = np.r_[X1[:,::2] / 15, X2]
        Y = np.r_[Y1, Y2]

        return X, Y

    else:
        NotImplementedError

# -------------------------------------------------------- UCI DATASETS

def fetch_GLASS(path, valid_size=0.2, test_size=0.2, rnd_state=1):

    path = Path(path)
    data_path = path / 'glass.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', data_path)

    data = np.genfromtxt(data_path, delimiter=',')
    
    X, Y = (data[:, 1:-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=rnd_state)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=rnd_state)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )