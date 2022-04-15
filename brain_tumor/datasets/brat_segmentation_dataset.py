"""
Brain Tumor Segmentation Dataset.

Author: Haoyi Zhu
"""

from __future__ import annotations

import os
from torch.utils.data import Dataset

from brain_tumor.utils import read_nib


class BraTSegmentationDataset(Dataset):
    """
    Dataset for brain tumor segmentation.

    Parameters
    ----------
    root: str
        Path to directory of segmentation data.
    train: bool
        Whether in training mode.
    img_dim: int, 2 or 3, default is 2
        The dimension of input images.
    mri_type: str, FLAIR or T1w or T1wCE or T2w
    """
    MRI_TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
    EXCLUDE_INDEXES = [109, 123, 709]

    def __init__(self, root, train=True, img_dim=2, mri_type='FLAIR'):
        super(BraTSegmentationDataset, self).__init__()
        self._root = os.path.join(root, 'BraTS2021_Training_Data')
        self._train = train
        self._img_dim = img_dim
        self._mri_type = mri_type
        assert self._mri_type in self.MRI_TYPES
        
        self._items, self._labels = self._prepare_data()

    def _prepare_data(self, labels=None):
        if self._img_dim == 2:
            return self._prepare_data_2d(labels)
        elif self._img_dim == 3:
            return self._prepare_data_3d(labels)
        else:
            raise NotImplementedError

    def _prepare_data_2d(self, labels=None):
        pass

    def _prepare_data_3d(self, labels=None):
        pass

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        pass