#2) ATTEND_GAN_Encoder
 # using pretrained ResNet-152(ImageNet dataset)
 # using Res5c layer to extract the spatial features
 # 7 * 7 * 2048 feature_map --> 49 * 2048 representing 49 semantic_based regions

import torch.nn as nn
from torchvision import models

device = 'cuda'
ResNet = models.resnet152(pretrained = True, progress = True)
Res5c = list(ResNet.children())[:-2]

class ATTEND_GAN_ResNet(nn.Module):
  def __init__(self):
    super(ATTEND_GAN_ResNet, self).__init__()
    self.feature_extraction_layer = nn.Sequential(*Res5c)
    
  def forward(self, image):
    feature_map = self.feature_extraction_layer(image)
    return feature_map

ATTEND_GAN_Encoder = ATTEND_GAN_ResNet().to(device)
