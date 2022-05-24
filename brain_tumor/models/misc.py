from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable


def spatial_broadcast(x, spatial_size: list[int]):
    """
    Broadcast action embedding or other coordinates along spatial dim.
    [N, d] -> [N, d, H, W] repeated over each HxW sptial location
    """
    sdim = len(spatial_size)
    x_unsqueeze = x.view(*x.size(), *(1,) * sdim)
    return x_unsqueeze.expand(*(-1,) * x.dim(), *spatial_size)


def spatial_broadcast_concat(feature_map, embedding):
    """
    Args:
        feature_map: [B, C, H, W]
        embedding: [B, d]
    """
    assert feature_map.dim() >= 3
    assert embedding.dim() == 2
    embedding = spatial_broadcast(embedding, feature_map.size()[2:])
    return torch.cat([feature_map, embedding], dim=1)


def get_activation(activation: str | Callable | None) -> Callable:
    if not activation:
        return nn.Identity
    elif callable(activation):
        return activation
    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "swish": lambda: nn.SiLU(inplace=True),  # SiLU is alias for Swish
        "sigmoid": nn.Sigmoid,
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]


def get_initializer(method: str | Callable, activation: str) -> Callable:
    if isinstance(method, str):
        assert hasattr(
            nn.init, f"{method}_"
        ), f"Initializer nn.init.{method}_ does not exist"
        if method == "orthogonal":
            try:
                gain = nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0
            return lambda x: nn.init.orthogonal_(x, gain=gain)
        else:
            return getattr(nn.init, f"{method}_")
    else:
        assert callable(method)
        return method
