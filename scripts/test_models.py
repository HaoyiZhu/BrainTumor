import brain_tumor.models.resnet_3d as r3
import brain_tumor.models.resnet_2d as r2
import brain_tumor.models.unet_2d as u2
import torch
import brain_tumor.models.vit_2d as v2
import brain_tumor.models.convnext_2d as c2
ccc=1
t2 = torch.ones((1, ccc, 512, 512))  # 一张 channel为ccc的512*512图片
t3 = torch.ones((1, 3, 64, 512, 512))

print(type(t2))
print(t2.shape)
print(t3.shape)
"""#res

mr2=r2.resnet_18_2d()
mr3=r3.resnet_18_3d(False,True)

print(mr3.forward(t3).shape)
print(mr2.forward(t2).shape)

"""
# unet

#mu2 = u2.UNet2D(3, 1, 64)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mr2=r2.resnet_101_2d()
#mvit=v2.my_vit().to(device)
print(mr2(t2).shape)
#print(mvit(t2.to(device)).shape)


mc2=c2.convnext_base()

print(mc2(t2).shape)
