import torch.nn as nn

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        upsample = True,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()
        
    #     self.scratch_layer1 = nn.Conv2d(
    #     features,
    #     features,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    #     bias=False,
    # )
        
        self.deconv = deconv
        self.align_corners = align_corners
        self.upsample = upsample

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if self.upsample == True:
            output = nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
            )

        output = self.out_conv(output)

        return output
    
class Scratch_layers(nn.Module):
    def __init__(self):
        super(Scratch_layers, self).__init__()
        
        self.layer1_rn = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        
        self.layer2_rn = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        
        self.layer3_rn = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        
    def forward(self, x1, x2, x3):
        x1 = self.layer1_rn(x1)
        x2 = self.layer2_rn(x2)
        x3 = self.layer3_rn(x3)
        
        return x1, x2, x3