"""
Brain Tumor Segmentation Dataset.

Author: Haoyi Zhu
"""

from __future__ import annotations

import os
from torch.utils.data import Dataset

import brain_tumor.utils as U


class BraTSegmentationDataset(Dataset):
    """
    Dataset for brain tumor segmentation.

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
        super(BraTSegmentationDataset, self).__init__()
        self._root = os.path.join(root, "BraTS2021_Training_Data")
        self._train = train
        self._img_dim = img_dim
        self._mri_type = mri_type
        self._check_mri_type()

        if self._img_dim == 2:
            from brain_tumor.utils.presets import SimpleTransform2D

            self.transformation = SimpleTransform2D()
        elif self._img_dim == 3:
            from brain_tumor.utils.presets import SimpleTransform3D

            self.transformation = SimpleTransform3D()
        else:
            raise NotImplementedError

        self._items = self._prepare_data()

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
        # label = self._labels[idx]

        img, label = self._load_img_3d(img_path)

        img, label = self.transformation(img, label)

        return img, label

    def _load_img(self, path):
        if self._img_dim == 2:
            return self._load_img_2d(labels)
        elif self._img_dim == 3:
            return self._load_img_3d(labels)
        else:
            raise NotImplementedError

    def _prepare_data(self, labels=None):
        dirs = os.listdir(self._root)
        dirs = [os.path.join(self._root, d) for d in dirs]

        return dirs

    def _load_img_3d(self, path):
        prefix = path.split('/')[-1]

        slices_list = []
        for mri_type in self._mri_type:
            slices = U.read_nib(
                os.path.join(path, f'{prefix}_{self._mri_type_to_file_name(mri_type)}.nii.gz')
            )
            slices_list.append(slices)

        img = U.slices_to_3d_img(slices_list)

        label = U.read_nib(
                os.path.join(path, f'{prefix}_seg.nii.gz')
            )

        return img, label

    def _load_img_2d(self, path):
        pass

    def _mri_type_to_file_name(self, mri_type):
        if mri_type == 'FLAIR':
            return 'flair'
        elif mri_type == 'T1wCE':
            return 't1ce'
        elif mri_type == 'T1w':
            return 't1'
        elif mri_type == 'T2w':
            return 't2'
        else:
            raise NotImplementedError


if __name__ == '__main__':
    dataset = BraTSegmentationDataset(
        root='/ssd3/Benchmark/haoyi/BRaTS2021/segmentation',
        mri_type=["T1wCE"], # "FLAIR", "T2w", "T1wCE"],
        )

    img, label = dataset.__getitem__(0)
    print(img.shape)
    print(label.shape)
