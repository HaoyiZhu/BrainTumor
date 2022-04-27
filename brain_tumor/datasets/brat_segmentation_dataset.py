"""
Brain Tumor Segmentation Dataset.

Author: Haoyi Zhu
"""

from __future__ import annotations

import os
import json
from omegaconf import DictConfig
from torch.utils.data import Dataset

import brain_tumor.utils as U


class BraTSegmentationDataset(Dataset):
    """
    Dataset for brain tumor segmentation.

    Parameters
    ----------
    cfg: dict
        Dataset configuration.
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
        cfg: DictConfig,
        root: str,
        train: bool = True,
        img_dim: int = 3,
        mri_type: str | list[str] = "T1wCE",
    ):
        super(BraTSegmentationDataset, self).__init__()
        self._cfg = cfg

        self._root = os.path.join(root, "BraTS2021_Training_Data")
        self._train = train
        self._img_dim = img_dim
        self._mri_type = mri_type
        self._check_mri_type()

        self._split = self._cfg.split

        if 'aug' in self._cfg:
            rot = self._cfg['aug']['rot_factor']
            rot_p = self._cfg['aug']['rot_p']
            scale_factor = self._cfg['aug']['scale_factor']
        else:
            rot = 0.
            rot_p = 0.
            scale_factor = 0.

        if self._img_dim == 2:
            from brain_tumor.utils.presets import SimpleTransform2D

            self.transformation = SimpleTransform2D(
                input_size=self._cfg.input_size,
                rot=rot,
                rot_p=rot_p,
                scale_factor=scale_factor,
                task='segmentation',
                train=self._train,)
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

        img, label = self._load_img(img_path)

        img, target, target_weight = self.transformation(img, label)

        return img, target, target_weight

    def _load_img(self, path):
        if self._img_dim == 2:
            return self._load_img_2d(path)
        elif self._img_dim == 3:
            return self._load_img_3d(path)
        else:
            raise NotImplementedError

    def _prepare_data(self):
        if self._img_dim == 2:
            return self._prepare_data_2d()
        elif self._img_dim == 3:
            return self._prepare_data_3d()

    def _prepare_data_2d(self):
        pass

    def _prepare_data_3d(self):
        items = []

        dirs = os.listdir(self._root)
        val_ids = json.load(
            open(
                f"{self._split.root}/val_ids_seed{self._split.seed}_ratio{self._split.ratio}.json",
                "r",
            )
        )

        for d in dirs:
            if d[:9] != "BraTS2021":
                continue
            data_id = d.split("_")[-1]
            if int(data_id) not in self.EXCLUDE_INDEXES:
                if (self._train and data_id not in val_ids) or (
                    not self._train and data_id in val_ids
                ):
                    items.append(os.path.join(self._root, d))

        return items

    def _load_img_3d(self, path):
        if "/" in path:
            prefix = path.split("/")[-1]
        else:
            prefix = path.split("\\")[-1]

        slices_list = []
        for mri_type in self._mri_type:
            slices = U.read_nib(
                os.path.join(
                    path, f"{prefix}_{self._mri_type_to_file_name(mri_type)}.nii.gz"
                )
            )
            slices_list.append(slices)

        img = U.slices_to_3d_img(slices_list)

        label = U.read_nib(os.path.join(path, f"{prefix}_seg.nii.gz"))

        return img, label

    def _load_img_2d(self, path):
        pass

    def _mri_type_to_file_name(self, mri_type):
        if mri_type == "FLAIR":
            return "flair"
        elif mri_type == "T1wCE":
            return "t1ce"
        elif mri_type == "T1w":
            return "t1"
        elif mri_type == "T2w":
            return "t2"
        else:
            raise NotImplementedError


if __name__ == "__main__":
    dataset = BraTSegmentationDataset(
        cfg=DictConfig(
            {"split": {"root": "./train_val_splits", "seed": 42, "ratio": 0.1}}
        ),
        root="/ssd3/Benchmark/haoyi/BRaTS2021/segmentation",
        mri_type=["T1wCE"],  # "FLAIR", "T2w", "T1wCE"],
    )

    img, label = dataset.__getitem__(0)
    print(img.shape)
    print(label.shape)
