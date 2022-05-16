"""
Criterion functions.

Author: Haoyi Zhu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    """
    Criterions for brain tumor tasks.
    """

    def __init__(
        self,
        task,
        **kwargs,
    ):
        super(Criterion, self).__init__()
        self.task = task

        self._build_loss_fn(**kwargs)

    def _build_loss_fn(self, **kwargs):
        if self.task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, output, target):
        if self.task == "classification":
            target = target.squeeze(-1)
        return self.loss_fn(output, target)
