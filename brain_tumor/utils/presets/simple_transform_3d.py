"""
Pre-process and data augmentation for 3d images.

Author: Haoyi Zhu
"""

import numpy as np
import torch


class SimpleTransform3D(object):
    """
    General class for 3d image pre-process and data augmantation.

    Parameters
    ----------
    """

    def __init__(
        self,
    ):
        pass

    def __call__(self, img, label):
        return img, label
