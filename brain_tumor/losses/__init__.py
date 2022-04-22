import torch
from .losses import *

names = {
    "CrossEntropy": torch.nn.CrossEntropyLoss,
}
