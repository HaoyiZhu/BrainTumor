"""
Pre-process and data augmentation for 2.5d images.

Author: Haoyi Zhu
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch

from brain_tumor.utils.transforms import im_to_torch, get_affine_transform


class SimpleTransform25D(object):
    """
    General class for 2d image pre-process and data augmantation.

    Parameters
    ----------
    input_size: tuple
        Input image size, as (height, width).
    rot: int
        Rotation augmentation.
    rot_p: float
        Probability of applying rotation augmentation.
    scale_factor: float
        Scale augmentation.
    task: str
        Specify classification or segementation task.
    train: bool
        True for training trasformation.
    """

    def __init__(
        self,
        input_size: tuple,
        rot: int,
        rot_p: float,
        scale_factor: float,
        task: str,
        train: bool,
    ) -> None:
        self._rot = rot
        self._rot_p = rot_p
        self._scale_factor = scale_factor
        self._task = task

        self._input_size = input_size
        self._train = train

    def _target_generator_classification(self, label):
        target = torch.LongTensor([label])
        target_weight = torch.LongTensor([1.0])

        return target, target_weight

    def _target_generator_segmentation(self, label):
        target = torch.FloatTensor([label])
        target_weight = torch.FloatTensor([1.0])

        return target, target_weight

    def __call__(self, srcs, label):
        # rescale
        if self._train:
            sf = self._scale_factor
            scale = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = 1.0

        # rotation
        if self._train:
            rf = self._rot
            r = (
                np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
                if random.random() <= self._rot_p
                else 0
            )
        else:
            r = 0

        inp_h, inp_w = self._input_size
        center = np.zeros((2), dtype=np.float32)
        center[0] = inp_w * 0.5
        center[1] = inp_h * 0.5
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])

        imgs = []

        for src in srcs:
            img = cv2.warpAffine(
                src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR
            )

            if len(img.shape) == 2:
                img = img[..., None]

            imgs.append(im_to_torch(img))

        if self._task == "classification":
            target, target_weight = self._target_generator_classification(label)
        elif self._task == "segmentation":
            target, target_weight = self._target_generator_segmentation(label)
        else:
            raise NotImplementedError

        return torch.stack(imgs), target, target_weight
