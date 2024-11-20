import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MobileNetV4Unet(nn.Module):
    """
    UNet, downsampling with MobileNetV4 backbone, upsampling only from stride 32 (level 5) up to stride 4 (level 2)
    """

    def __init__(
        self,
        backbone="mobilenetv4_conv_small",
        mode="bilinear",
        n_classes=80,
        pretrained=True,
        width_scale=0.5,
    ):
        super(MobileNetV4Unet, self).__init__()
        self.n_classes = n_classes
        self.mode = mode

        # Define the backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )

        # Get in_channels at different stages
        if "mobilenetv4_conv_small" in backbone:
            in_channels_list = [32, 32, 64, 96, 960]
        elif "mobilenetv4_conv_medium" in backbone:
            in_channels_list = [32, 48, 80, 160, 960]
        elif "mobilenetv4_conv_large" in backbone:
            in_channels_list = [24, 48, 96, 192, 960]
        else:
            raise ValueError("Unsupported backbone")

        factor = 1 if mode == "convtranspose" else 2

        # Apply width_scale to the channel counts
        base_channels = int(64 * width_scale)

        # Convolution layers to adjust channels from backbone
        self.conv5 = nn.Conv2d(
            in_channels_list[4], base_channels * 16 // factor, kernel_size=1
        )
        self.conv4 = nn.Conv2d(in_channels_list[3], base_channels * 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels_list[2], base_channels * 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels_list[1], base_channels * 2, kernel_size=1)
        # self.conv1 = nn.Conv2d(in_channels_list[0], base_channels, kernel_size=1)

        # Decoder path with skip connections
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, mode)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, mode)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, mode)
        # self.up4 = Up(base_channels * 2, base_channels, mode)

        # Final output convolution
        self.outc = OutConv(base_channels * 2 // factor, n_classes)

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)  # List of feature maps

        # Assign features and apply 1x1 convolutions to adjust channels
        # x1 = self.conv1(features[0])  # High resolution, low-level features
        x2 = self.conv2(features[1])
        x3 = self.conv3(features[2])
        x4 = self.conv4(features[3])
        x5 = self.conv5(features[4])  # Low resolution, high-level features

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # x = self.up4(x, x1)

        # Final output
        logits = self.outc(x)
        return logits


class UNet(nn.Module):
    def __init__(self, n_classes, mode="bilinear", width_scale=1, in_channels=3):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.mode = mode

        # Apply width_scale to the channel counts
        base_channels = int(64 * width_scale)

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        factor = 1 if mode == "convtranspose" else 2

        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, mode)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, mode)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, mode)
        self.up4 = Up(base_channels * 2, base_channels, mode)

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mode="bilinear"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        elif mode == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode=mode)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        elif mode == "convtranspose":
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            raise ValueError("mode must be 'bilinear' or 'nearest' or 'convtranspose'")

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # NOTE: padding is not needed if the input_size is divisible by 64 (so that spatial size of outputs in all stride from 2 to 32 are even number)
        # NOTE: Converter1 do NOT work with the following padding code
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    # Test the model
    # model = UNet(n_classes=1, mode="bilinear", width_scale=1, in_channels=3)
    # model2 = UNet(n_classes=1, mode="convtranspose", width_scale=1, in_channels=3)

    # x = torch.randn(1, 3, 224, 224)

    # print(f"mode=bilinear, output: {model(x).shape}")
    # print(f"mode=convtranspose, output: {model2(x).shape}")

    model = MobileNetV4Unet(n_classes=80, backbone="mobilenetv4_conv_small")
    model2 = MobileNetV4Unet(
        n_classes=80, backbone="mobilenetv4_conv_medium", mode="convtranspose"
    )
    x = torch.randn(1, 3, 320, 320)
    print(f"MobileNetV4Unet output: {model(x).shape}")
    print(f"MobileNetV4Unet output: {model2(x).shape}")
