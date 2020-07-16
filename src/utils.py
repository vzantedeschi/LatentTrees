import os

import numpy as np
import requests

from tqdm import tqdm
import torch

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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

def iterate_minibatches(*tensors, batch_size, shuffle=True, epochs=1, allow_incomplete=True, callback=lambda x:x):
    indices = np.arange(len(tensors[0]))
    upper_bound = int((np.ceil if allow_incomplete else np.floor) (len(indices) / batch_size)) * batch_size
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start: batch_start + batch_size]
            batch = [tensor[batch_ix] for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        
        self.x = X
        self.y = Y
        print(X.shape, Y.shape)

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]