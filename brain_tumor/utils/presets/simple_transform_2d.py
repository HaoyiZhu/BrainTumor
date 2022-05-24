"""
Pre-process and data augmentation for 2d images.

Author: Haoyi Zhu
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import PIL
import torch
from torchvision import transforms

from brain_tumor.utils.transforms import im_to_torch, get_affine_transform


class SimpleTransform2D(object):
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
        h_flip_p: float,
        task: str,
        train: bool,
    ) -> None:
        self._rot = rot
        self._rot_p = rot_p
        self._scale_factor = scale_factor
        self._h_flip_p = h_flip_p
        self._task = task

        self._input_size = input_size
        self._train = train

        if self._train:
            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=self._h_flip_p),
                    transforms.RandomRotation(
                        degrees=rot_p, resample=PIL.Image.BILINEAR
                    ),
                    transforms.RandomChoice(
                        [
                            transforms.Resize(self._input_size),
                            transforms.RandomResizedCrop(
                                self._input_size,
                                scale=(1 - self._scale_factor, 1),
                            ),
                        ],
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(self._input_size),
                    transforms.ToTensor(),
                ]
            )

    def _target_generator_classification(self, label):
        target = torch.tensor([label])
        target_weight = torch.tensor([1.0])

        return target, target_weight

    def _target_generator_segmentation(self, label):
        target = torch.FloatTensor([label])
        target_weight = torch.FloatTensor([1.0])

        return target, target_weight

    def __call__(self, src, label):
        img = self._transform(src)

        if self._task == "classification":
            target, target_weight = self._target_generator_classification(label)
        elif self._task == "segmentation":
            target, target_weight = self._target_generator_segmentation(label)
        else:
            raise NotImplementedError

        return img, target, target_weight
