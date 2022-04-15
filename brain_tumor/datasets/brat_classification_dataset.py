"""
Brain Tumor Radiogenomic Classification Dataset.

Author: Haoyi Zhu
"""

from __future__ import annotations

import os
from torch.utils.data import Dataset

from brain_tumor.utils import read_csv, read_dicom


class BraTClassificationDataset(Dataset):
    """
    Dataset for brain tumor radiogenomic classification.

    Parameters
    ----------
    root: str
        Path to directory of classification data.
    train: bool
        Whether in training mode.
    img_dim: int, 2 or 3, default is 2
        The dimension of input images.
    mri_type: str or list of str, default is T1wCE
        The MRI type of input images. Candidates include FLAIR, T1w, T2w and T1wCE
    """

    MRI_TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
    EXCLUDE_INDEXES = [109, 123, 709]

    def __init__(
        self,
        root: str,
        train: bool = True,
        img_dim: int = 2,
        mri_type: str | list[str] = "T1wCE",
    ):
        super(BraTClassificationDataset, self).__init__()
        self._root = root
        self._train = train
        self._img_dim = img_dim
        self._mri_type = mri_type
        self._check_mri_type()

        self._img_dir = os.path.join(root, "train" if self._train else "test")

        labels = None
        if self._train:
            labels = read_csv(os.path.join(self._root, "train_labels.csv"))

        self._items, self._labels = self._prepare_data(labels)

    def _check_mri_type(self):
        if isinstance(self._mri_type, list):
            assert set(self._mri_type) <= set(
                self.MRI_TYPES
            ), f"Wrong MRI type: {self._mri_type}"
        elif isinstance(self._mri_type, str):
            assert self._mri_type in self.MRI_TYPES, f"Wrong MRI type: {self._mri_type}"
        else:
            raise NotImplementedError

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
