import numpy as np
import gzip
import shutil
import tarfile

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils import download

# -------------------------------------------------------- CLUSTERING DATASET FETCHERS

def fetch_GLASS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'glass.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', data_path)

    data = np.genfromtxt(data_path, delimiter=',')
    
    X, Y = (data[:, 1:-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_COVTYPE(path, valid_size=0.2, test_size=0.2, seed=None):

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_ALOI(path, valid_size=0.2, test_size=0.2, seed=None):

    from PIL import Image
    from tqdm import tqdm

    path = Path(path)
    data_path = path / 'aloi_red4'
    npz_path = path / 'aloi_red4.npz'

    if not data_path.exists():

        path.mkdir(parents=True, exist_ok=True)

        for data_type in ["ill", "col", "view", "stereo"]:
            
            archive_path = path / f'aloi_red4_{data_type}.tar'

            download(f'http://aloi.science.uva.nl/tars/aloi_red4_{data_type}.tar', archive_path)

            with tarfile.open(archive_path, 'r') as f_in:

                f_in.extractall(path=data_path)

    if not npz_path.exists():
        
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

        np.savez_compressed(npz_path, X=X, Y=Y)
    
    else:

        data = np.load(npz_path)
        X, Y = data['X'], data['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_DIGITS(path, valid_size=0.2, test_size=0.2, seed=None):

    from sklearn.datasets import load_digits

    X, Y = load_digits(return_X_y=True)
    X, Y = X.reshape(-1, 1, 8, 8).astype(np.uint8), Y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )
