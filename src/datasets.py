from pathlib import Path

from sklearn.preprocessing import QuantileTransformer

from src.clus_datasets import *
from src.tabular_datasets import *
from src.toy_datasets import *

REAL_DATASETS = {
    'A9A': fetch_A9A,
    'EPSILON': fetch_EPSILON,
    'PROTEIN': fetch_PROTEIN,
    'YEAR': fetch_YEAR,
    'HIGGS': fetch_HIGGS,
    'MICROSOFT': fetch_MICROSOFT,
    'YAHOO': fetch_YAHOO,
    'CLICK': fetch_CLICK,
    'GLASS': fetch_GLASS,
    'COVTYPE': fetch_COVTYPE,
    'ALOI': fetch_ALOI,
    'DIGITS': fetch_DIGITS,
    'MUSH': fetch_MUSHROOMS,
    'TTT': fetch_TICTACTOE,
}

TOY_DATASETS = [
    'xor',
    'reg-xor',
    'swissroll',
]

class Dataset:

    """
    Code adapted from https://github.com/Qwicen/node/blob/master/lib/data.py .

    """
    def __init__(self, dataset, data_path='./DATA', normalize=False, normalize_target=False, quantile_transform=False, quantile_noise=1e-3, in_features=None, out_features=None, flatten=False, **kwargs):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATASETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
        :param quantile_transform: whether tranform the feature distributions into normals, using a quantile transform
        :param quantile_noise: magnitude of the quantile noise
        :param in_features: which features to use as inputs
        :param out_features: which features to reconstruct as output
        :param flatten: whether flattening instances to vectors
        :param kwargs: depending on the dataset, you may select train size, test size or other params
        """

        if dataset in REAL_DATASETS:
            data_dict = REAL_DATASETS[dataset](Path(data_path) / dataset, **kwargs)

            self.X_train = data_dict['X_train']
            self.y_train = data_dict['y_train']
            self.X_valid = data_dict['X_valid']
            self.y_valid = data_dict['y_valid']
            self.X_test = data_dict['X_test']
            self.y_test = data_dict['y_test']

            if flatten:
                self.X_train, self.X_valid, self.X_test = self.X_train.reshape(len(self.X_train), -1), self.X_valid.reshape(len(self.X_valid), -1), self.X_test.reshape(len(self.X_test), -1)

            if normalize:

                print("Normalize dataset")
                axis = [0] + [i + 2 for i in range(self.X_train.ndim - 2)]
                self.mean = np.mean(self.X_train, axis=tuple(axis), dtype=np.float32)
                self.std = np.std(self.X_train, axis=tuple(axis), dtype=np.float32)

                # if constants, set std to 1
                self.std[self.std == 0.] = 1.

                if dataset not in ['ALOI']:
                    self.X_train = (self.X_train - self.mean) / self.std
                    self.X_valid = (self.X_valid - self.mean) / self.std
                    self.X_test = (self.X_test - self.mean) / self.std

            if quantile_transform:
                quantile_train = np.copy(self.X_train)
                if quantile_noise:
                    stds = np.std(quantile_train, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_train += noise_std * np.random.randn(*quantile_train.shape)

                qt = QuantileTransformer(output_distribution='normal').fit(quantile_train)
                self.X_train = qt.transform(self.X_train)
                self.X_valid = qt.transform(self.X_valid)
                self.X_test = qt.transform(self.X_test)

            if normalize_target:

                print("Normalize target value")
                self.mean_y = np.mean(self.y_train, axis=0, dtype=np.float32)
                self.std_y = np.std(self.y_train, axis=0, dtype=np.float32)

                # if constants, set std to 1
                if self.std_y == 0.:
                    self.std_y = 1.

                self.y_train = (self.y_train - self.mean_y) / self.std_y
                self.y_valid = (self.y_valid - self.mean_y) / self.std_y
                self.y_test = (self.y_test - self.mean_y) / self.std_y

            if in_features is not None:
                self.X_train_in, self.X_valid_in, self.X_test_in = self.X_train[:, in_features], self.X_valid[:, in_features], self.X_test[:, in_features]

            if out_features is not None:
                self.X_train_out, self.X_valid_out, self.X_test_out = self.X_train[:, out_features], self.X_valid[:, out_features], self.X_test[:, out_features]

        elif dataset in TOY_DATASETS:
            data_dict = toy_dataset(distr=dataset, **kwargs)

            self.X = data_dict['X']
            self.Y = data_dict['Y']
            if 'labels' in data_dict:
                self.labels = data_dict['labels']

        self.data_path = data_path
        self.dataset = dataset

class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, *data, **options):
        
        n_data = len(data)
        if n_data == 0:
            raise ValueError("At least one set required as input")

        self.data = data
        means = options.pop('means', None)
        stds = options.pop('stds', None)
        self.transform = options.pop('transform', None)
        self.test = options.pop('test', False)

        if options:
            raise TypeError("Invalid parameters passed: %s" % str(options))
         
        if means is not None:
            assert stds is not None, "must specify both <means> and <stds>"

            self.normalize = lambda data: [(d - m) / s for d, m, s in zip(data, means, stds)]

        else:
            self.normalize = lambda data: data

    def __len__(self):

        return len(self.data[0])

    def __getitem__(self, idx):

        data = self.normalize([s[idx] for s in self.data])

        if self.transform:

            if self.test:
                data = sum([[self.transform.test_transform(d)] * 2 for d in data], [])
            else:
                data = sum([self.transform(d) for d in data], [])
            
        return data