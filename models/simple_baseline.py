import torch
import torch.nn as nn
import timm
from typing import Type


class MobilenetV4SimpleBaseline(nn.Module):
    def __init__(
        self,
        backbone="mobilenetv4_conv_medium.e500_r256_in1k",
        n_classes=1,
        mode="bilinear",
    ):
        super(MobilenetV4SimpleBaseline, self).__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        if "small" in backbone:
            inter_channel = 192
            out_channel = 64
        elif "medium" in backbone:
            inter_channel = 256
            out_channel = 128
        else:  # "large" in backbone
            inter_channel = 256
            out_channel = 256

        if mode == "convtranspose":
            inter_channel *= 2
            out_channel *= 2

        self.conv11 = torch.nn.Conv2d(960, inter_channel, 1)
        if mode == "bilinear" or mode == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode=mode)
            self.conv1 = ConvBlock(inter_channel, inter_channel)
            self.conv2 = ConvBlock(inter_channel, out_channel)
            self.conv3 = ConvBlock(out_channel, n_classes, act_layer=None)
        elif mode == "convtranspose":
            self.up = None
            self.deconv1 = DeconvBlock(
                inter_channel, out_channel, kernel_size=4, padding=1, stride=2
            )
            self.deconv2 = DeconvBlock(
                out_channel, out_channel, kernel_size=4, padding=1, stride=2
            )
            self.deconv3 = DeconvBlock(
                out_channel,
                n_classes,
                kernel_size=4,
                padding=1,
                stride=2,
                act_layer=None,
            )
        else:
            raise ValueError("mode must be 'bilinear', 'nearest', or 'convtranspose'")

    def forward(self, x):
        x = self.model.forward_features(
            x
        )  # output is unpooled (1, 960, H/32, W/32) shaped tensor
        if self.up:  # use nn.Upsample
            return self.conv3(
                self.up(self.conv2(self.up(self.conv1(self.up(self.conv11(x))))))
            )
        else:  # use nn.ConvTranspose2d
            return self.deconv3(self.deconv2(self.deconv1(self.conv11(x))))


class ConvBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        cardinality: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            kernel_size: Kernel size for convolution layers.
            padding: Padding used in convolution layers.
            stride: Stride used in convolution layers.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            dilation: Dilation rate for convolution layers.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super(ConvBlock, self).__init__()

        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock does not support changing base width"

        self.conv1 = nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = norm_layer(outplanes)
        if act_layer:
            self.act1 = act_layer(inplace=True)
        else:
            self.act1 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        cardinality: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            kernel_size: Kernel size for convolution layers.
            padding: Padding used in convolution layers.
            stride: Stride used in convolution layers.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            dilation: Dilation rate for convolution layers.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super(DeconvBlock, self).__init__()

        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock does not support changing base width"

        self.conv1 = nn.ConvTranspose2d(
            inplanes,
            outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = norm_layer(outplanes)
        if act_layer:
            self.act1 = act_layer(inplace=True)
        else:
            self.act1 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
