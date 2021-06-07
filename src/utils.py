import numpy as np

import random
import requests
import os

import torch
from tqdm import tqdm

def deterministic(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():

    if torch.cuda.is_available():
        pin_memory = True
        device = torch.device("cuda:0")

    else:
        pin_memory = False
        device = torch.device("cpu")

    return pin_memory, device

"""
Code adapted from https://github.com/Qwicen/node/blob/master/lib/data.py .

"""

def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """ saves file from url to filename with a fancy progressbar """
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))

    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename

# --------------------------------------------------------------- COMPOSITION FUNCTIONS

def concat_func(x, z):

    return torch.cat((x.view(len(x), -1), z), 1)

def freezed_concat_func(x, z):
    """ used to remove skip connection """

    zeros = torch.zeros_like(x)

    return torch.cat((zeros.view(len(x), -1), z), 1)

def none_func(x, z):

    return z

# --------------------------------------------------------------- BUILDING BLOCK MODULES

class MLP(torch.nn.Module):
    
    def __init__(self, in_size, out_size, layers=2, dropout=0., **kwargs):
        
        super(MLP, self).__init__()
        
        if layers == 1:
            units = [(in_size, out_size)]
        else:
            units = [(in_size, 64)]
            for i in range(1, layers - 1):
                units.append((units[-1][1], units[-1][1] * 2))
            units.append((units[-1][1], out_size))
        
        self.layers = []
        for i, u in enumerate(units):
            self.layers.append(torch.nn.Linear(*u))
            
            if i < layers - 1: # end the model with a linear layer
                self.layers.append(torch.nn.ELU())
                
                if dropout > 0.:
                    self.layers.append(torch.nn.Dropout(dropout))
        
        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        
        return self.net(x)

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, in_size, out_size, linear, **kwargs):
        
        super(LogisticRegression, self).__init__()

        if linear:
            self.net = torch.nn.Linear(in_size, out_size)
        else:
            self.net = MLP(in_size, out_size, **kwargs)   

    def forward(self, x):

        y_pred = torch.sigmoid(self.net(x))
        
        return y_pred


# ----------------------------------------------------------------------- SPLIT MODULES

class Split(torch.nn.Module):

    def __init__(self, out_size):

        super(Split, self).__init__()

        self.split = torch.nn.Sequential(*self.layers, torch.nn.BatchNorm1d(out_size))

    def forward(self, x):

        return self.split(x)

class ConvSplit(Split):

    def __init__(self, in_size, out_size, **kwargs):
            
        self.in_size = in_size
        
        if in_size == (1, 28, 28): # fashion MNIST

            self.layers = [
                torch.nn.BatchNorm2d(1),
                torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(32, 64, kernel_size=3),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64*6*6, 600),
                torch.nn.Dropout2d(0.25),
                torch.nn.Linear(600, 120),
                torch.nn.Linear(120, out_size),
            ]

        else:
            raise NotImplementedError

        super(ConvSplit, self).__init__(out_size)

class LinearSplit(Split):

    def __init__(self, in_size, out_size, **kwargs):

        self.in_size = np.prod(in_size)

        self.layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(self.in_size, out_size)
        ]

        super(LinearSplit, self).__init__(out_size)

class MLPSplit(Split):

    def __init__(self, in_size, out_size, **kwargs):

        self.in_size = np.prod(in_size)

        self.num_layers = kwargs.pop("split_layers", 2)
        self.dropout = kwargs.pop("split_dropout", 0.)

        self.layers = [
            torch.nn.Flatten(),
            *MLP(self.in_size, out_size, layers=self.num_layers, dropout=self.dropout).layers
        ]

        super(MLPSplit, self).__init__(out_size)

split_dict = { # supported Split modules
    "conv": ConvSplit,
    "linear": LinearSplit,
    "mlp": MLPSplit,
}

act_dict = { # supported activation functions
    "none": torch.nn.Identity(),
    "elu": torch.nn.ELU(),
    "tanh": torch.nn.Tanh(),
}
