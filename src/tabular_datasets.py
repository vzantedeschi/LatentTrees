"""
    Code adapted from https://github.com/Qwicen/node/blob/master/lib/data.py .

"""
import os
import bz2
import numpy as np
import pandas as pd
import gzip
import shutil

import torch
import warnings

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from category_encoders import LeaveOneOutEncoder
from category_encoders.ordinal import OrdinalEncoder
from pathlib import Path

from src.utils import download

def fetch_A9A(path, train_size=None, valid_size=None, test_size=None, **kwargs):
    train_path = os.path.join(path, 'a9a')
    test_path = os.path.join(path, 'a9a.t')
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/9cqdx166iwonrj9/a9a?dl=1", train_path)
        download("https://www.dropbox.com/s/sa0ds895c0v4xc6/a9a.t?dl=1", test_path)

    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=123)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=123)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/xy4wwvutwikmtha/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/nthpxofymrais5s/stratified_test_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]    

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
    )


def fetch_EPSILON(path, train_size=None, valid_size=None, test_size=None, **kwargs):
    train_path = os.path.join(path, 'epsilon_normalized')
    test_path = os.path.join(path, 'epsilon_normalized.t')
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        train_archive_path = os.path.join(path, 'epsilon_normalized.bz2')
        test_archive_path = os.path.join(path, 'epsilon_normalized.t.bz2')
        if not all(os.path.exists(fname) for fname in (train_archive_path, test_archive_path)):
            download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2", train_archive_path)
            download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2", test_archive_path)
        print("unpacking dataset")
        for file_name, archive_name in zip((train_path, test_path), (train_archive_path, test_archive_path)):
            zipfile = bz2.BZ2File(archive_name)
            with open(file_name, 'wb') as f:
                f.write(zipfile.read())

    print("reading dataset (it may take a long time)")
    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=2000)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=2000)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/wxgm94gvm6d3xn5/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/fm4llo5uucdglti/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
    )


def fetch_PROTEIN(path, train_size=None, valid_size=None, test_size=None, **kwargs):
    """
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#protein
    """
    train_path = os.path.join(path, 'protein')
    test_path = os.path.join(path, 'protein.t')
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/pflp4vftdj3qzbj/protein.tr?dl=1", train_path)
        download("https://www.dropbox.com/s/z7i5n0xdcw57weh/protein.t?dl=1", test_path)
    for fname in (train_path, test_path):
        raw = open(fname).read().replace(' .', '0.')
        with open(fname, 'w') as f:
            f.write(raw)

    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=357)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=357)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/wq2v9hl1wxfufs3/small_stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/7o9el8pp1bvyy22/small_stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
    )


def fetch_YEAR(path, train_size=None, valid_size=None, test_size=51630, **kwargs):
    data_path = os.path.join(path, 'data.csv')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1', data_path)
    n_features = 91
    types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
    data = pd.read_csv(data_path, header=None, dtype=types)
    data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    if all(sizes is None for sizes in (train_size, valid_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/00u6cnj9mthvzj1/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/420uhjvjab1bt7k/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test,
    )


def fetch_HIGGS(path, train_size=None, valid_size=None, test_size=5 * 10 ** 5, **kwargs):
    data_path = os.path.join(path, 'higgs.csv')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'HIGGS.csv.gz')
        download('https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', archive_path)
        with gzip.open(archive_path, 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    n_features = 29
    types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
    data = pd.read_csv(data_path, header=None, dtype=types)
    data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    if all(sizes is None for sizes in (train_size, valid_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/i2uekmwqnp9r4ix/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/wkbk74orytmb2su/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test,
    )


def fetch_MICROSOFT(path, **kwargs):
    train_path = os.path.join(path, 'msrank_train.tsv')
    test_path = os.path.join(path, 'msrank_test.tsv')
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/izpty5feug57kqn/msrank_train.tsv?dl=1", train_path)
        download("https://www.dropbox.com/s/tlsmm9a6krv0215/msrank_test.tsv?dl=1", test_path)

        for fname in (train_path, test_path):
            raw = open(fname).read().replace('\\t', '\t')
            with open(fname, 'w') as f:
                f.write(raw)

    data_train = pd.read_csv(train_path, header=None, skiprows=1, sep='\t')
    data_test = pd.read_csv(test_path, header=None, skiprows=1, sep='\t')

    train_idx_path = os.path.join(path, 'train_idx.txt')
    valid_idx_path = os.path.join(path, 'valid_idx.txt')
    if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
        download("https://www.dropbox.com/s/pba6dyibyogep46/train_idx.txt?dl=1", train_idx_path)
        download("https://www.dropbox.com/s/yednqu9edgdd2l1/valid_idx.txt?dl=1", valid_idx_path)
    train_idx = pd.read_csv(train_idx_path, header=None)[0].values
    valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values

    X_train, y_train, query_train = data_train.iloc[train_idx, 2:].values, data_train.iloc[train_idx, 0].values, data_train.iloc[train_idx, 1].values
    X_valid, y_valid, query_valid = data_train.iloc[valid_idx, 2:].values, data_train.iloc[valid_idx, 0].values, data_train.iloc[valid_idx, 1].values
    X_test, y_test, query_test = data_test.iloc[:, 2:].values, data_test.iloc[:, 0].values, data_test.iloc[:, 1].values

    return dict(
        X_train=X_train.astype(np.float32), y_train=y_train.astype(np.int64), query_train=query_train,
        X_valid=X_valid.astype(np.float32), y_valid=y_valid.astype(np.int64), query_valid=query_valid,
        X_test=X_test.astype(np.float32), y_test=y_test.astype(np.int64), query_test=query_test,
    )


def fetch_YAHOO(path, *args):
    train_path = os.path.join(path, 'yahoo_train.tsv')
    valid_path = os.path.join(path, 'yahoo_valid.tsv')
    test_path = os.path.join(path, 'yahoo_test.tsv')
    if not all(os.path.exists(fname) for fname in (train_path, valid_path, test_path)):
        os.makedirs(path, exist_ok=True)
        train_archive_path = os.path.join(path, 'yahoo_train.tsv.gz')
        valid_archive_path = os.path.join(path, 'yahoo_valid.tsv.gz')
        test_archive_path = os.path.join(path, 'yahoo_test.tsv.gz')
        if not all(os.path.exists(fname) for fname in (train_archive_path, valid_archive_path, test_archive_path)):
            download("https://www.dropbox.com/s/7rq3ki5vtxm6gzx/yahoo_set_1_train.gz?dl=1", train_archive_path)
            download("https://www.dropbox.com/s/3ai8rxm1v0l5sd1/yahoo_set_1_validation.gz?dl=1", valid_archive_path)
            download("https://www.dropbox.com/s/3d7tdfb1an0b6i4/yahoo_set_1_test.gz?dl=1", test_archive_path)

        for file_name, archive_name in zip((train_path, valid_path, test_path), (train_archive_path, valid_archive_path, test_archive_path)):
            with gzip.open(archive_name, 'rb') as f_in:
                with open(file_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        for fname in (train_path, valid_path, test_path):
            raw = open(fname).read().replace('\\t', '\t')
            with open(fname, 'w') as f:
                f.write(raw)

    data_train = pd.read_csv(train_path, header=None, skiprows=1, sep='\t')
    data_valid = pd.read_csv(valid_path, header=None, skiprows=1, sep='\t')
    data_test = pd.read_csv(test_path, header=None, skiprows=1, sep='\t')

    X_train, y_train, query_train = data_train.iloc[:, 2:].values, data_train.iloc[:, 0].values, data_train.iloc[:, 1].values
    X_valid, y_valid, query_valid = data_valid.iloc[:, 2:].values, data_valid.iloc[:, 0].values, data_valid.iloc[:, 1].values
    X_test, y_test, query_test = data_test.iloc[:, 2:].values, data_test.iloc[:, 0].values, data_test.iloc[:, 1].values

    return dict(
        X_train=X_train.astype(np.float32), y_train=y_train, query_train=query_train,
        X_valid=X_valid.astype(np.float32), y_valid=y_valid, query_valid=query_valid,
        X_test=X_test.astype(np.float32), y_test=y_test, query_test=query_test,
    )


def fetch_CLICK(path, valid_size=100_000, seed=None, **kwargs):
    # based on: https://www.kaggle.com/slamnz/primer-airlines-delay
    csv_path = os.path.join(path, 'click.csv')
    if not os.path.exists(csv_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1', csv_path)

    data = pd.read_csv(csv_path, index_col=0)
    X, y = data.drop(columns=['target']), data['target']
    X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
    y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()

    y_train = (y_train.values.reshape(-1) == 1).astype('int64')
    y_test = (y_test.values.reshape(-1) == 1).astype('int64')

    cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                    'keyword_id', 'title_id', 'description_id', 'user_id']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=seed)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_val[cat_features] = cat_encoder.transform(X_val[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])
    return dict(
        X_train=X_train.values.astype('float32'), y_train=y_train,
        X_valid=X_val.values.astype('float32'), y_valid=y_val,
        X_test=X_test.values.astype('float32'), y_test=y_test
    )

def fetch_MUSHROOMS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'agaricus-lepiota.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', data_path)

    data = pd.read_csv(data_path, names=np.arange(23))
    encoder = OrdinalEncoder(return_df=False)
    data = encoder.fit_transform(data)
    
    X, Y = (data[:, 1:]).astype(np.float32), (data[:, 0] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_TICTACTOE(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'tic-tac-toe.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', data_path)

    data = pd.read_csv(data_path, names=np.arange(10))
    encoder = OrdinalEncoder(return_df=False)
    data = encoder.fit_transform(data)
    
    X, Y = (data[:, :-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )