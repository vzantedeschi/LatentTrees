from pathlib import Path

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
    def __init__(self, dataset, data_path='./DATA', normalize=False, normalize_target=False, in_features=None, out_features=None, **kwargs):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATSETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
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

            if normalize:

                print("Normalize dataset")
                if self.X_train.ndim == 2:
                    self.mean = np.mean(self.X_train, axis=0, dtype=np.float32)
                    self.std = np.std(self.X_train, axis=0, dtype=np.float32)

                else:
                    self.mean = np.mean(self.X_train, axis=(0, 2, 3), dtype=np.float32)
                    self.std = np.std(self.X_train, axis=(0, 2, 3), dtype=np.float32)

                # if constants, set std to 1
                self.std[self.std == 0.] = 1.

                if dataset != 'ALOI':
                    self.X_train = (self.X_train - self.mean) / self.std
                    self.X_valid = (self.X_valid - self.mean) / self.std
                    self.X_test = (self.X_test - self.mean) / self.std

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

    def __init__(self, data, means=None, stds=None, device=None, transform=None):
        
        self.data = data # a list of sets, e.g. (X, Y)
        self.device = device
        self.transform = transform
        
        if means is not None:
            assert stds is not None, "must specify both <means> and <stds>"

            self.normalize = lambda data: [(d - m) / s for d, m, s in zip(data, means, stds)]

        else:
            self.normalize = lambda data: data

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        data = self.normalize([s[idx] for s in self.data])

        if self.device:
            data = [torch.from_numpy(d).to(self.device) for d in data]

        if self.transform:
            data = [self.transform(d) for d in data]
            
        return x, y
