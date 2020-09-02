import numpy as np

from pathlib import Path

import h5py
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
    data_path = path / 'aloi_red4'
    hdf_path = path / f'aloi_red4-rnd={rnd_state}.hdf5'

    if not data_path.exists():

        path.mkdir(parents=True, exist_ok=True)

        for data_type in ["ill", "col", "view", "stereo"]:
            
            archive_path = path / f'aloi_red4_{data_type}.tar'

            download(f'http://aloi.science.uva.nl/tars/aloi_red4_{data_type}.tar', archive_path)

            with tarfile.open(archive_path, 'r') as f_in:

                f_in.extractall(path=data_path)

    if not hdf_path.exists():
        
        X = np.empty((110250, 3, 144, 192), dtype=np.uint8)
        Y = np.empty(110250, dtype=np.uint16)

        # loop over classes
        i = 0
        for c in tqdm(range(1000), desc='Converting ALOI png to npy'):

            c_path = data_path / "png4" / str(c + 1) 

            # loop over class instances
            for i_path in c_path.glob('*.png'):
                
                im_frame = Image.open(i_path)
                X[i] = np.transpose(np.array(im_frame), (2, 0, 1))
                Y[i] = c
                i += 1

        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=rnd_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=rnd_state)

        with h5py.File(hdf_path, 'w') as f:
            f.create_dataset("X_train", data=X_train, compression='gzip')
            f.create_dataset("X_val", data=X_val, compression='gzip')
            f.create_dataset("X_test", data=X_test, compression='gzip')
            f.create_dataset("y_train", data=y_train, compression='gzip')
            f.create_dataset("y_val", data=y_val, compression='gzip')
            f.create_dataset("y_test", data=y_test, compression='gzip')
    
    else:

        f = h5py.File(hdf_path, 'r')
        X_train, y_train = f['X_train'], f['y_train']
        X_val, y_val = f['X_val'], f['y_val']
        X_test, y_test = f['X_test'], f['y_test']

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )
