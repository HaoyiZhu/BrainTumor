from .resnet_3d import (
    resnet_18_3d,
    resnet_34_3d,
    resnet_50_3d,
    resnet_101_3d,
    resnet_152_3d,
    resnext_50_32x4d_3d,
    resnext_101_32x8d_3d,
)


names = {
    "resnet_18_3d": resnet_18_3d,
    "resnet_34_3d": resnet_34_3d,
    "resnet_50_3d": resnet_50_3d,
    "resnet_101_3d": resnet_101_3d,
    "resnet_152_3d": resnet_152_3d,
    "resnext_50_32x4d_3d": resnext_50_32x4d_3d,
    "resnext_101_32x8d_3d": resnext_101_32x8d_3d,
}
