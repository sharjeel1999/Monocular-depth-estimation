import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# import torchvision

class ResnetEncoder_pre(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder_pre, self).__init__()

        model = models.detection.maskrcnn_resnet50_fpn(weights = (models.detection.MaskRCNN_ResNet50_FPN_Weights))
        # self.transform = model.transform
        
        # self.transform = transforms.Compose([
        #         transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        #     ])
        
        self.backbone = model.backbone

    def forward(self, input_image):
        # print('before image shape: ', input_image.shape)
        # input_image = self.transform(input_image)
        # print('after image shape: ', input_image.shape)

        self.features = []
        base_out = self.backbone(input_image)
        
        # print(base_out['pool'].shape)
        # print(base_out['3'].shape)
        # print(base_out['2'].shape)
        # print(base_out['1'].shape)
        # print(base_out['0'].shape)
        
        self.features.append(base_out['0'])
        self.features.append(base_out['1'])
        self.features.append(base_out['2'])
        self.features.append(base_out['3'])
        self.features.append(base_out['pool'])

        return self.features