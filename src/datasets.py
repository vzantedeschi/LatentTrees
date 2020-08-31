import numpy as np

from pathlib import Path

import gzip
import shutil
import tarfile

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

def fetch_COVTYPE(path, valid_size=0.2, test_size=0.2, rnd_state=1):

    path = Path(path)
    data_path = path / 'covtype.data'

    if not data_path.exists():
        path.mkdir(parents=True)
        archive_path = path / 'covtype.data.gz'
        download('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz', archive_path)

        with gzip.open(archive_path, 'rb') as f_in:

            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    data = np.genfromtxt(data_path, delimiter=',')
    
    X, Y = (data[:, :-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=rnd_state)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=rnd_state)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_ALOI(path, valid_size=0.2, test_size=0.2, rnd_state=1):

    from PIL import Image
    from tqdm import tqdm

    path = Path(path)
    data_path = path / f'aloi_red4'

    if not data_path.exists():

        path.mkdir(parents=True, exist_ok=True)

        for data_type in ["ill", "col", "view", "stereo"]:
            
            archive_path = path / f'aloi_red4_{data_type}.tar'

            download(f'http://aloi.science.uva.nl/tars/aloi_red4_{data_type}.tar', archive_path)

            with tarfile.open(archive_path, 'r') as f_in:

                f_in.extractall(path=data_path)

    X, Y = np.empty((0, 144, 192, 3)), np.empty((0)) 
    # loop over classes
    for c in tqdm(range(1000), desc='Converting ALOI png to npy'):

        c_path = data_path / "png4" / str(c + 1) 

        # loop over class instances
        for i_path in c_path.glob('*.png'):

            im_frame = Image.open(i_path)
            X = np.append(X, np.array(im_frame)[None, :], axis=0)
            Y = np.append(Y, [c], axis=0)
        
    X = np.transpose(X.astype(np.float32), (0, 3, 1, 2))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=rnd_state)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=rnd_state)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )