from __future__ import annotations
import torch.nn as nn
from typing import Callable
from .misc import get_activation, get_initializer

__all__ = ["build_mlp"]


def build_mlp(
    input_dim,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int,
    activation: str | Callable = "relu",
    weight_init: str | Callable = "orthogonal",
    bias_init="zeros",
    norm_layer: str | None = "batchnorm",
    output_activation: str | None = None,
):
    """
    In other popular RL implementations, tanh is typically used with orthogonal
    initialization, which may perform better than ReLU.
    """
    act_layer = get_activation(activation)

    weight_init = get_initializer(weight_init, activation)
    bias_init = get_initializer(bias_init, activation)

    if norm_layer is not None:
        norm_layer = norm_layer.lower()

    if not norm_layer:
        norm_layer = nn.Identity
    elif norm_layer == "batchnorm":
        norm_layer = nn.BatchNorm1d
    elif norm_layer == "layernorm":
        norm_layer = nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm layer: {norm_layer}")

    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), norm_layer(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))

    if output_activation:
        mods += [norm_layer(output_dim), get_activation(output_activation)()]

    for mod in mods:
        if isinstance(mod, nn.Linear):
            weight_init(mod.weight)
            bias_init(mod.bias)

    return nn.Sequential(*mods)
