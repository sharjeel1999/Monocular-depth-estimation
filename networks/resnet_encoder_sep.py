import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# import torchvision

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class conv_block(nn.Module):
    def __init__(self, in_channels):
        super(conv_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace = True)
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResnetEncoder_student(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder_student, self).__init__()

        model = models.detection.maskrcnn_resnet50_fpn(weights = (models.detection.MaskRCNN_ResNet50_FPN_Weights)) # can also use resnet 101 and resnet 152
        # print(model)
        # self.transform = model.transform
        
        # self.transform = transforms.Compose([
        #         transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        #     ])
        
        self.backbone = model.backbone
        # print(self.backbone)
        self.backbone.body.conv1 = Identity()
        
        self.buffer_m1 = conv_block(256)
        self.buffer_m2 = conv_block(256)
        

    def forward(self, input_image):
        # print('before image shape: ', input_image.shape)
        # input_image = self.transform(input_image)
        # print('after image shape: ', input_image.shape)

        self.features = []
        base_out = self.backbone(input_image)
        
        print('pool: ', base_out['pool'].shape)
        print('3: ', base_out['3'].shape)
        print('2: ', base_out['2'].shape)
        print('1: ', base_out['1'].shape)
        print('0: ', base_out['0'].shape)
        
        self.features.append(base_out['0'])
        self.features.append(base_out['1'])
        self.features.append(base_out['2'])
        self.features.append(base_out['3'])
        self.features.append(base_out['pool'])
        
        # buffered_m1 = self.buffer_m1(base_out['pool'])
        # buffered_m2 = self.buffer_m2(base_out['3'])
        
        return self.features
    

class Initial_student(nn.Module):
    def __init__(self):
        super(Initial_student, self).__init__()
        
        self.initial_layer = nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
        
    def forward(self, input_image):
        return self.initial_layer(input_image)
