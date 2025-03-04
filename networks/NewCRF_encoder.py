import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace = True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class CRF_Encoder_student(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(CRF_Encoder_student, self).__init__()
        
        version = 'large07'
        max_depth = 10
        window_size = int(version[-2:])
        pretrain = None
        frozen_stages=-1
        norm_cfg = dict(type='BN', requires_grad=True)
        
        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )
        
        self.backbone = SwinTransformer(**backbone_cfg)
        
        self.backbone.patch_embed = Identity()
        
        self.sync1 = conv_block(192, 256)
        self.sync2 = conv_block(384, 256)
        self.sync3 = conv_block(768, 256)
        self.sync4 = conv_block(1536, 256)
        
        self.mp = nn.MaxPool2d(2)
        self.sync5 = conv_block(1536, 256)
        

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
        
        a0 = base_out[0] #F.interpolate(base_out[0], [96, 320], mode="bilinear", align_corners=False)
        a1 = base_out[1] #F.interpolate(base_out[1], [48, 160], mode="bilinear", align_corners=False)
        a2 = base_out[2] #F.interpolate(base_out[2], [24, 80], mode="bilinear", align_corners=False)
        a3 = base_out[3] #F.interpolate(base_out[3], [12, 40], mode="bilinear", align_corners=False)
        
        self.features.append(self.sync1(a0))
        self.features.append(self.sync2(a1))
        self.features.append(self.sync3(a2))
        self.features.append(self.sync4(a3))
        
        zz = self.mp(base_out[3])
        # print('zz shape: ', zz.shape)
        self.features.append(self.sync5(zz))
        
        # buffered_m1 = self.buffer_m1(base_out['pool'])
        # buffered_m2 = self.buffer_m2(base_out['3'])
        
        return self.features
    


class CRF_Initial_student(nn.Module):
    def __init__(self):
        super(CRF_Initial_student, self).__init__()
        
        version = 'large07'
        max_depth = 10
        window_size = int(version[-2:])
        pretrain = None
        frozen_stages=-1
        norm_cfg = dict(type='BN', requires_grad=True)
        
        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )
        
        self.backbone = SwinTransformer(**backbone_cfg)
        self.initial_layer = self.backbone.patch_embed
        
        
    def forward(self, input_image):
        return self.initial_layer(input_image)



class CRF_Initial_teacher(nn.Module):
    def __init__(self):
        super(CRF_Initial_teacher, self).__init__()
        
        # version = 'large07'
        # max_depth = 10
        # window_size = int(version[-2:])
        # pretrain = None
        # frozen_stages=-1
        # norm_cfg = dict(type='BN', requires_grad=True)
        
        # window_size = int(version[-2:])

        # if version[:-2] == 'base':
        #     embed_dim = 128
        #     depths = [2, 2, 18, 2]
        #     num_heads = [4, 8, 16, 32]
        #     in_channels = [128, 256, 512, 1024]
        # elif version[:-2] == 'large':
        #     embed_dim = 192
        #     depths = [2, 2, 18, 2]
        #     num_heads = [6, 12, 24, 48]
        #     in_channels = [192, 384, 768, 1536]
        # elif version[:-2] == 'tiny':
        #     embed_dim = 96
        #     depths = [2, 2, 6, 2]
        #     num_heads = [3, 6, 12, 24]
        #     in_channels = [96, 192, 384, 768]

        # backbone_cfg = dict(
        #     embed_dim=embed_dim,
        #     depths=depths,
        #     num_heads=num_heads,
        #     window_size=window_size,
        #     ape=False,
        #     drop_path_rate=0.3,
        #     patch_norm=True,
        #     use_checkpoint=False,
        #     frozen_stages=frozen_stages
        # )

        
        # self.backbone = SwinTransformer(**backbone_cfg)
        # self.initial_layer = self.backbone.patch_embed
        # self.initial_layer.proj.in_channels = 28
        
        self.initial_layer = nn.Sequential(
                nn.Conv2d(28, 192, kernel_size = (4, 4), stride = (4, 4)),
                # nn.LayerNorm([12, 192, 48, 160], eps = 1e-05, elementwise_affine = True)
            )
        
    def set_layer_norm(self, x):
        b, c, h, w = x.shape
        self.ln = nn.LayerNorm([c, h, w], eps = 1e-05, elementwise_affine = True).cuda()
        
    def forward(self, input_image):
        # print(self.initial_layer)
        # print('input shape: ', input_image.shape)
        x = self.initial_layer(input_image)
        self.set_layer_norm(x)
        x = self.ln(x)
        # print('output shape: ', x.shape)
        return x