import numpy as np
import pandas as pd
import pydicom

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