"""
Brain Tumor Radiogenomic Classification Dataset.

Author: Haoyi Zhu
"""

from __future__ import annotations

import os
from torch.utils.data import Dataset

import brain_tumor.utils as U


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
        img_dim: int = 3,
        mri_type: str | list[str] = "T1wCE",
    ):
        super(BraTClassificationDataset, self).__init__()
        self._root = root
        self._train = train
        self._img_dim = img_dim
        self._mri_type = mri_type
        self._check_mri_type()

        self._img_dir = os.path.join(root, "train" if self._train else "test")

        if self._img_dim == 2:
            from brain_tumor.utils.presets import SimpleTransform2D

            self.transformation = SimpleTransform2D()
        elif self._img_dim == 3:
            from brain_tumor.utils.presets import SimpleTransform3D

            self.transformation = SimpleTransform3D()
        else:
            raise NotImplementedError

        self._items, self._labels = self._prepare_data()

    def _check_mri_type(self):
        if isinstance(self._mri_type, list):
            assert set(self._mri_type) <= set(
                self.MRI_TYPES
            ), f"Wrong MRI type: {self._mri_type}"
        elif isinstance(self._mri_type, str):
            assert self._mri_type in self.MRI_TYPES, f"Wrong MRI type: {self._mri_type}"
            self._mri_type = [self._mri_type]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]

        img = self._load_img(img_path)

        img, label = self.transformation(img, label)

        return img, label

    def _load_img(self, path):
        if self._img_dim == 2:
            return self._load_img_2d(path)
        elif self._img_dim == 3:
            return self._load_img_3d(path)
        else:
            raise NotImplementedError

    def _prepare_data(self):
        annotations = U.read_csv(os.path.join(self._root, "train_labels.csv"))

        items = [
            os.path.join(self._img_dir, str(data_id).zfill(5)) for data_id in annotations["BraTS21ID"]
        ]

        labels = [int(ann) for ann in annotations["MGMT_value"]]

        return items, labels

    def _load_img_3d(self, path):
        slices_list, ids_list = [], []
        for mri_type in self._mri_type:
            slices, ids = U.read_dicom_dir(
                os.path.join(path, mri_type)
            )
            slices_list.append(slices)
            ids_list.append(ids)

        img = U.slices_to_3d_img(slices_list, ids_list)

        return img

    def _load_img_2d(self, path):
        pass


if __name__ == '__main__':
    dataset = BraTClassificationDataset(
        root='/ssd3/Benchmark/haoyi/BRaTS2021/classification',
        mri_type=["FLAIR"], # , "T2w", "T1wCE"],
        )

    img, label = dataset.__getitem__(0)
    print(img.shape)
    print(label)
