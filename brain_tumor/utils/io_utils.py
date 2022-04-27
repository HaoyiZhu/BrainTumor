"""
IO related utils.
"""

import os

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from scipy import ndimage


def read_csv(filename):
    return pd.read_csv(filename)


def read_dicom(filename, normalize=False):
    img = pydicom.read_file(filename).pixel_array

    if normalize:
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)

        img = (img * 255).astype(np.uint8)

    return img


def read_nib(filename):
    return np.asarray(nib.load(filename).dataobj)


def read_dicom_dir(path, normalize=False):
    slices = []
    idxes = []

    for fn in os.listdir(path):
        slices.append(read_dicom(os.path.join(path, fn), normalize=normalize))
        idxes.append(int(fn.split(".")[0].split("-")[-1]))

    return slices, idxes


def slices_to_3d_img(slices, ids=None):
    return np.concatenate([np.array(s) for s in slices], axis=0)


def resize_volume(img):
    """
    Resize 3d image across z-axis.

    Parameters
    ----------
    img: numpy.ndarray
        Numpy array in shape (W, H, D)
    """
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
