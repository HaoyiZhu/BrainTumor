from .resnet_2d import (
    resnet_18_2d,
    resnet_34_2d,
    resnet_50_2d,
    resnet_101_2d,
    resnet_152_2d,
)

from .resnet_3d import (
    resnet_18_3d,
    resnet_34_3d,
    resnet_50_3d,
    resnet_101_3d,
    resnet_152_3d,
    resnext_50_32x4d_3d,
    resnext_101_32x8d_3d,
)

from .mlp import build_mlp
from .convnext_2d import(
    convnext_tiny,
    convnext_base,
    convnext_small,
    convnext_large,
    convnext_xlarge,
)

from .vit_2d import(
    vit_base,
)

names = {
    "resnet_18_2d": resnet_18_2d,
    "resnet_34_2d": resnet_34_2d,
    "resnet_50_2d": resnet_50_2d,
    "resnet_101_2d": resnet_101_2d,
    "resnet_152_2d": resnet_152_2d,
    "resnet_18_3d": resnet_18_3d,
    "resnet_34_3d": resnet_34_3d,
    "resnet_50_3d": resnet_50_3d,
    "resnet_101_3d": resnet_101_3d,
    "resnet_152_3d": resnet_152_3d,
    "resnext_50_32x4d_3d": resnext_50_32x4d_3d,
    "resnext_101_32x8d_3d": resnext_101_32x8d_3d,
    "mlp": build_mlp,
    "convnext_tiny": convnext_tiny,
    "convnext_base": convnext_base,
    "convnext_small": convnext_small,
    "convnext_large": convnext_large,
    "convnext_xlarge": convnext_xlarge,
    "vit_base": vit_base,
}
