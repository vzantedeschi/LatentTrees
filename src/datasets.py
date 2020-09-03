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
    def __init__(self, dataset, data_path='./DATA', normalize=False, in_features=None, out_features=None, **kwargs):
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
                self.mean = np.mean(self.X_train, axis=0, dtype=np.float32)
                self.std = np.std(self.X_train, axis=0, dtype=np.float32)

                # if constants, set std to 1
                self.std[self.std == 0.] = 1.

                if dataset != 'ALOI':
                    self.X_train = (self.X_train - self.mean) / self.std
                    self.X_valid = (self.X_valid - self.mean) / self.std
                    self.X_test = (self.X_test - self.mean) / self.std

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

    def __init__(self, X, Y, means=None, stds=None, device=None):
        
        self.x = X
        self.y = Y
        self.device = None
        print(X.shape, Y.shape)
        
        if means is not None:
            assert stds is not None, "must specify both <means> and <stds>"

            self.normalize = lambda x, y: ((x - means[0]) / stds[0], (y - means[1]) / stds[1])

        else:
            self.normalize = lambda x, y: (x, y)

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        x, y = self.normalize(self.x[idx], self.y[idx])

        if self.device:
            return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

        else:
            return x, y