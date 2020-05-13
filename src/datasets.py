import glob
import math
import numpy as np
import os

import netCDF4 as nc4

from sklearn.datasets import make_swiss_roll

import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------------- PREPROCESSING

class Normalizer(object):

    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, image):

        return (image - self.mean) / self.std

class Scaler(object):

    def __init__(self, min_values, max_values, dim=(256, 256)):
        """ scales to [0,1] range """

        self.min_values = np.repeat(min_values, np.prod(dim)).reshape(len(min_values), *dim)
        self.max_values = np.repeat(max_values, np.prod(dim)).reshape(len(max_values), *dim)

    def __call__(self, image):

        return (image - self.min_values) / (self.max_values - self.min_values)

def compute_normalizer(dataloader, dim=13):

    mean = torch.zeros(dim)
    std = torch.zeros(dim)

    nb_instances = 0

    # estimate mean
    for batch in dataloader:
        mean += torch.sum(batch["radiances"], 0)
        nb_instances += len(batch["radiances"])

    mean /= nb_instances
    # estimate std
    for batch in dataloader:
        std += torch.sum((batch["radiances"] - mean)**2, 0)

    std = (std / (nb_instances - 1))**0.5

    return Normalizer(mean.numpy().reshape(-1, 1, 1), std.numpy().reshape(-1, 1, 1))

def compute_scaler(dataloader, dim=13):

    min_values = torch.Tensor([float("Inf")] * dim)
    max_values = torch.Tensor([-float("Inf")] * dim)

    for batch in dataloader:

        c_min = torch.min(batch["radiances"], 0)[0]
        c_max = torch.max(batch["radiances"], 0)[0]
        
        min_values = torch.min(c_min, min_values)
        max_values = torch.max(c_max, max_values)

    return Scaler(min_values.numpy(), max_values.numpy())

# ------------------------------------------------------------ TOY DATASETS

def toy_dataset(n=1000, distr="xor", dim=2):

    if distr == "xor":
        
        X = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n, dim))
        Y = (X[:,0] * X[:,1] >= 0).astype(int)

    elif distr == "normal":

        n2 = n // 2
        X1 = np.random.normal((0,-2), (1,2), (n2, dim))
        Y1 = np.ones(n2)

        X2 = np.random.normal((0,2), (1,2), (n2, dim))
        Y2 = np.zeros(n2)

        X = np.r_[X1, X2]
        Y = np.r_[Y1, Y2]

    elif distr == "swissroll":

        n2 = n // 2

        X1,_ = make_swiss_roll(n_samples=n2, noise=0)
        Y1 = np.ones(n2)

        X2 = np.random.uniform(low=tuple([-1.] * dim), high=tuple([1.] * dim), size=(n2, dim))
        Y2 = np.zeros(n2)

        X = np.r_[X1[:,::2] / 15, X2]
        Y = np.r_[Y1, Y2]

    return X,Y

# ------------------------------------------------------------ CUMULO HELPERS

radiances = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33', 'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26', 'ev_1km_emissive_27', 'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
coordinates = ['latitude', 'longitude']
properties = ['cloud_water_path', 'cloud_optical_thickness', 'cloud_effective_radius', 'cloud_phase_optical_properties', 'cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature', 'cloud_emissivity', 'surface_temperature']
rois = 'cloud_mask'
labels = 'cloud_layer_type'

property_ranges = np.array([[0, 0, 0, 0, 10, 0, -150, 0, -150], [5000, 100, 100, 4, 800, 18000, 50, 1, 50]])

def class_pixel_collate(batch, label=7):
    """ collate batch instances by stacking their pixels. Select only cloudy pixels of class <label>."""
    
    res = {'radiances': [], 'properties': []}

    for instance in batch:

        # select pixels belonging to class <label> that are cloudy and have valid property values
        class_pixels = instance['labels'][0] == label
        cloudy_pixels = instance['rois'][0] == 1
        valid_pixels = np.sum(np.isnan(instance['properties']), 0) == 0

        mask = cloudy_pixels & class_pixels & valid_pixels
        nb_pixels = np.sum(mask)

        for key, value in res.items():
            value.append(instance[key].transpose(1, 2, 0)[mask].reshape(nb_pixels, -1))

    return {key: torch.from_numpy(np.vstack(value)).float() for key, value in res.items()}

def get_most_frequent_label(labelmask, dim=0):

    labels = np.argmax(labelmask, 0).astype(float)

    # set label of pixels with no occurences of clouds to NaN
    labels[np.sum(labelmask, 0) == 0] = np.NaN

    return labels

def read_nc(nc_file):
    """return masked arrays, with masks indicating the invalid values"""
    
    file = nc4.Dataset(nc_file, 'r', format='NETCDF4')

    f_radiances = np.vstack([file.variables[name][:] for name in radiances])
    f_properties = np.vstack([file.variables[name][:] for name in properties])
    f_rois = file.variables[rois][:]
    f_labels = file.variables[labels][:]

    return f_radiances, f_properties, f_rois, f_labels

def read_npz(npz_file):

    file = np.load(npz_file)

    return file['radiances'], file['properties'], file['cloud_mask'], file['labels']

class CumuloDataset(Dataset):

    def __init__(self, root_dir, rad_preproc=None, prop_preproc=None, ext="npz", prop_idx=None):
        
        self.root_dir = root_dir
        self.ext = ext

        self.file_paths = glob.glob(os.path.join(root_dir, "*." + ext))

        if len(self.file_paths) == 0:
            raise FileNotFoundError("no " + ext + " files in", self.root_dir)

        self.rad_preproc = rad_preproc
        self.prop_preproc = prop_preproc
        self.prop_idx = prop_idx

    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, idx):

        filename = self.file_paths[idx]
        
        if self.ext == "npz":
            radiances, properties, rois, labels = read_npz(filename)

        elif self.ext == "nc":
            radiances, properties, rois, labels = read_nc(filename)

        if self.rad_preproc is not None:
            radiances = self.rad_preproc(radiances)

        if self.prop_idx is not None:
            properties = properties[self.prop_idx]

        if self.prop_preproc is not None:
            properties = self.prop_preproc(properties)

        return {"filename": filename, "radiances": radiances, "properties": properties, "rois": rois, "labels": labels}

    def __str__(self):
        return 'CUMULO'