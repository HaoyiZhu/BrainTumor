"""
Brain Tumor Radiogenomic Classification Dataset.

Author: Haoyi Zhu
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

import brain_tumor.utils as U


class SNNClassificationDataset(Dataset):
    """
    Dataset for brain tumor radiogenomic classification.

    Parameters
    ----------
    exp_specs: dict
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
        exp_specs: DictConfig,
        root: str,
        train: bool = True,
        img_dim: int | float = 2,
        mri_type: str | list[str] = "T1wCE",
    ):
        super(SNNClassificationDataset, self).__init__()
        self._exp_specs = exp_specs

        self._root = root
        self._train = train
        self._img_dim = img_dim
        self._mri_type = mri_type
        self._check_mri_type()

        self._split = self._exp_specs["split"]
        if self._img_dim == 2.5:
            self._max_slice_num = self._exp_specs["max_slice_num"]

        # self._img_dir = os.path.join(root, "train" if self._train else "test")
        self._img_dir = os.path.join(root, "train")

        if self._train and "aug" in self._exp_specs:
            rot = self._exp_specs["aug"]["rot_factor"]
            rot_p = self._exp_specs["aug"]["rot_p"]
            scale_factor = self._exp_specs["aug"]["scale_factor"]
            h_flip_p = self._exp_specs["aug"]["h_flip_p"]
        else:
            rot = 0.0
            rot_p = 0.0
            scale_factor = 0.0
            h_flip_p = 0.0

        if self._img_dim == 2:
            from brain_tumor.utils.presets import SimpleTransform2D

            self.transformation = SimpleTransform2D(
                input_size=self._exp_specs["input_size"],
                rot=rot,
                rot_p=rot_p,
                scale_factor=scale_factor,
                h_flip_p=h_flip_p,
                task="classification",
                train=self._train,
            )
        elif self._img_dim == 2.5:
            from brain_tumor.utils.presets import SimpleTransform25D

            self.transformation = SimpleTransform25D(
                input_size=self._exp_specs["input_size"],
                rot=rot,
                rot_p=rot_p,
                scale_factor=scale_factor,
                task="classification",
                train=self._train,
            )
        elif self._img_dim == 3:
            from brain_tumor.utils.presets import SimpleTransform3D

            self.transformation = SimpleTransform3D()
        else:
            raise NotImplementedError

        self._items, self._labels = self._prepare_data()

    def _check_mri_type(self):
        if isinstance(self._mri_type, ListConfig):
            self._mri_type = list(self._mri_type)

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

        img, target, target_weight = self.transformation(img, label)
        target = target.squeeze()

        return img, target

    def _load_img(self, path):
        if self._img_dim == 2:
            return self._load_img_2d(path)
        elif self._img_dim == 2.5:
            return self._load_img_25d(path)
        elif self._img_dim == 3:
            return self._load_img_3d(path)
        else:
            raise NotImplementedError

    def _prepare_data(self):
        if self._img_dim == 2:
            return self._prepare_data_2d()
        elif self._img_dim == 2.5 or self._img_dim == 3:
            return self._prepare_data_gt2d()

    def _prepare_data_2d(self):
        items, labels = [], []
        root = self._split["root"]
        seed = self._split["seed"]
        ratio = self._split["ratio"]
        val_ids = json.load(open(f"{root}/val_ids_seed{seed}_ratio{ratio}.json", "r",))
        annotations = U.read_csv(os.path.join(self._root, "train_labels.csv"))

        for i, data_id in enumerate(annotations["BraTS21ID"]):
            if data_id not in self.EXCLUDE_INDEXES:
                data_id_str = str(data_id).zfill(5)
                if (self._train and data_id_str not in val_ids) or (
                    not self._train and data_id_str in val_ids
                ):
                    for mri_type in self._mri_type:
                        parent_img_path = os.path.join(
                            self._img_dir, data_id_str, mri_type
                        )
                        for img_path in os.listdir(parent_img_path):
                            items.append(os.path.join(parent_img_path, img_path))
                            labels.append(int(annotations["MGMT_value"][i]))

        return items, labels

    def _prepare_data_gt2d(self):
        # greater than 2d (2.5d and 3d)
        items, labels = [], []

        val_ids = json.load(
            open(
                f"{self._split['root']}/val_ids_seed{self._split['seed']}_ratio{self._split['ratio']}.json",
                "r",
            )
        )
        annotations = U.read_csv(os.path.join(self._root, "train_labels.csv"))

        for i, data_id in enumerate(annotations["BraTS21ID"]):
            if data_id not in self.EXCLUDE_INDEXES:
                data_id_str = str(data_id).zfill(5)
                if (self._train and data_id_str not in val_ids) or (
                    not self._train and data_id_str in val_ids
                ):
                    items.append(os.path.join(self._img_dir, data_id_str))
                    labels.append(int(annotations["MGMT_value"][i]))

        return items, labels

    def _load_img_3d(self, path):
        slices_list, ids_list = [], []
        for mri_type in self._mri_type:
            slices, ids = U.read_3d_dicom_dir(
                os.path.join(path, mri_type), normalize=True
            )
            slices_list.append(slices)
            ids_list.append(ids)

        img = U.slices_to_3d_img(slices_list, ids_list)

        return img

    def _load_img_2d(self, path):
        slices = U.read_2d_dicom_dir(path, normalize=True)
        img = U.slices_to_2d_img(slices)

        return img

    def _load_img_25d(self, path):
        # load images for 2.5d
        assert isinstance(self._mri_type, str) or (
            isinstance(self._mri_type, list) and len(self._mri_type) == 1
        ), f"2.5d mode only support single mri type, but got {self._mri_type}."

        sub_path = (
            self._mri_type if isinstance(self._mri_type, str) else self._mri_type[0]
        )
        path = os.path.join(path, sub_path)

        imgs = []
        for fn in os.listdir(path):
            slices = U.read_2d_dicom_dir(os.path.join(path, fn), normalize=True)
            img = U.slices_to_2d_img(slices)
            imgs.append(img)

        num_slices = len(imgs)

        if self._train and num_slices > self._max_slice_num:
            start_slice_idx = np.random.randint(0, num_slices - self._max_slice_num)
            imgs = imgs[start_slice_idx : start_slice_idx + self._max_slice_num]

        # if self._train and num_slices < self._max_slice_num:
        #     imgs = np.tile(imgs, (self._max_slice_num // num_slices + 1, 1, 1, 1))

        return imgs

    def _collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        targets = torch.stack(targets)

        if self._img_dim == 2.5:

            imgs = [item.numpy() for item in imgs]
            imgs = torch.tensor(np.array(imgs)).view(
                -1,
                1,
                1,
                self._exp_specs["input_size"][0],
                self._exp_specs["input_size"][1],
            )

            while imgs.shape[0] < self._max_slice_num:
                imgs = torch.cat(
                    [imgs, imgs[0 : self._max_slice_num - imgs.shape[0]]], dim=0
                )

            return imgs, targets
        else:
            imgs = torch.stack(imgs)

            return imgs, targets
