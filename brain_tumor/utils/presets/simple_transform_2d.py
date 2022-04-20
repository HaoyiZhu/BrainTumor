"""
Pre-process and data augmentation for 2d images.

Author: Haoyi Zhu
"""

import numpy as np
import torch


class SimpleTransform2D(object):
    """
    General class for 2d image pre-process and data augmantation.

    Parameters
    ----------
    """

    def __init__(
        self,
    ):
        pass

    def __call__(self, img, label):
        return img, label
