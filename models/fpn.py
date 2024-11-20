import torch
import torch.nn as nn
import timm
from mmseg.models.necks.fpn import FPN
from mmdet.models.necks.pafpn import PAFPN
from mmseg.models.decode_heads.fpn_head import FPNHead
from mmseg.models.utils import Upsample
import numpy as np
from transformers import MobileNetV2Config, MobileNetV2Model

from models.mobilenetv4 import MobileNetV4, UniversalInvertedBottleneckBlock


class OCDFPN(torch.nn.Module):
    def __init__(
        self,
        backbone="mobilenetv4_conv_medium.e500_r256_in1k",
        n_classes=1,
        num_outs=4,
        out_channel=32,  # output channel of FPN neck
        dropout_ratio=0.1,
        fpn_type="mm",
    ):
        super(OCDFPN, self).__init__()
        self.pretrained = True
        ## get in_channels of the backbone
        # mobilenetv4
        if "mobilenetv4_conv_small" in backbone:
            in_channels = [32, 32, 64, 96, 960]
        elif backbone == "mobilenetv4_conv_medium_seg":
            in_channels = [32, 32, 80, 160, 448]
        elif ("mobilenetv4_conv_medium" in backbone) or (
            "mobilenetv4_hybrid_medium" in backbone
        ):
            in_channels = [32, 48, 80, 160, 960]
        elif "mobilenetv4_conv_large" in backbone:
            in_channels = [24, 48, 96, 192, 960]
        # mobilenetv1
        elif "mobilenetv1_100" in backbone:
            in_channels = [64, 128, 256, 512, 1024]
        # mobilenetv2
        elif "mobilenetv2_050" in backbone:
            in_channels = [8, 16, 16, 48, 160]
        elif "mobilenetv2_075" in backbone:
            in_channels = [16, 24, 24, 72, 240]
            self.pretrained = False
        elif "mobilenetv2_100" in backbone:
            in_channels = [16, 24, 32, 96, 320]
        elif "mobilenetv2_140" in backbone:
            in_channels = [24, 32, 48, 136, 448]
        # mobilenetv3
        elif "mobilenetv3_large_100" in backbone:
            in_channels = [16, 24, 40, 112, 960]
        elif "mobilenetv3_large_075" in backbone:
            in_channels = [16, 24, 32, 88, 720]
            self.pretrained = False
        elif "mobilenetv3_small_100" in backbone:
            in_channels = [16, 16, 24, 48, 576]
        # mobileone
        elif "mobileone_s0" in backbone:
            in_channels = [48, 48, 128, 256, 1024]
        elif "mobileone_s1" in backbone:
            in_channels = [64, 96, 192, 512, 1280]
        elif "mobileone_s2" in backbone:
            in_channels = [64, 96, 256, 640, 2048]
        elif "mobileone_s3" in backbone:
            in_channels = [64, 128, 320, 768, 2048]
        elif "mobileone_s4" in backbone:
            in_channels = [64, 192, 448, 896, 2048]
        # convnext
        elif ("convnext_tiny" in backbone) or ("convnext_small" in backbone):
            in_channels = [96, 192, 384, 768]
        # resnet
        elif "resnet18" in backbone or "resnet34" in backbone:
            in_channels = [64, 64, 128, 256, 512]
        elif "resnet50" in backbone or "resnet101" in backbone:
            in_channels = [64, 256, 512, 1024, 2048]
        # repvgg
        elif "repvgg_a0" in backbone:
            in_channels = [48, 48, 96, 192, 1280]
        elif "repvgg_a1" in backbone:
            in_channels = [64, 64, 128, 256, 1280]
        elif "repvgg_a2" in backbone:
            in_channels = [64, 96, 192, 384, 1408]
        elif "repvgg_b1" in backbone:  # for b1 and b1g4
            in_channels = [64, 128, 256, 512, 2048]
        # efficientnet
        elif "efficientnet_b0" in backbone:
            in_channels = [16, 24, 40, 112, 320]
        elif "efficientnet_b3" in backbone:
            in_channels = [24, 32, 48, 136, 384]
        elif "efficientnet_b5" in backbone:
            in_channels = [24, 40, 64, 176, 512]
        # efficientnetv2
        elif "efficientnetv2_rw_t" in backbone:
            in_channels = [24, 40, 48, 128, 208]
        elif "efficientnetv2_rw_s" in backbone:
            in_channels = [24, 48, 64, 160, 272]
        # hrnet
        elif "hrnet_w18" in backbone:
            in_channels = [64, 128, 256, 512, 1024]
        # efficientvit
        elif "efficientvit_b2" in backbone:
            in_channels = [48, 96, 192, 384]
        else:
            raise ValueError("backbone not supported")

        if backbone == "mobilenetv4_conv_medium_seg":
            self.backbone = MobileNetV4(
                arch="medium_seg", n_classes=-1, width_multiplier=1.0
            )
        else:
            # Load the pretrained MobileNetV4 model as the backbone
            self.backbone = timm.create_model(
                backbone, pretrained=self.pretrained, features_only=True
            )

        # only feed last num_outs channels to FPN
        # self.out_channel = self.in_channels[0]
        self.out_channel = out_channel
        self.num_outs = num_outs
        self.in_channels = in_channels[-self.num_outs :]
        self.out_channels = [self.out_channel for _ in range(self.num_outs)]
        if n_classes < 24:
            self.head_last_channel = in_channels[0]
        else:
            if "v4_conv_small" in backbone:
                self.head_last_channel = in_channels[0]
            elif "v4_conv_medium" in backbone:
                self.head_last_channel = 96
            elif "v4_conv_large" in backbone:
                self.head_last_channel = 128
            else:
                self.head_last_channel = 80
        # backbone_outputs always has 5 elements, but we only want the last num_outs elements
        if len(in_channels) == 5:
            self.feature_strides = [2, 4, 8, 16, 32][-self.num_outs :]
            self.in_index = [0, 1, 2, 3, 4][: self.num_outs]
        elif len(in_channels) == 4:
            self.feature_strides = [2, 4, 8, 16][-self.num_outs :]
            self.in_index = [0, 1, 2, 3][: self.num_outs]
        else:
            raise ValueError(f"in_channels length {len(in_channels)} not supported")

        # define fpn internal module type
        if fpn_type == "mm":
            self.neck_cls = FPN
            self.head_cls = FPNHead
        elif fpn_type == "extra_dw":
            self.neck_cls = FPNExtraDW
            self.head_cls = FPNHeadExtraDW
        elif fpn_type == "ib":
            self.neck_cls = FPNIB
            self.head_cls = FPNHeadIB
        elif fpn_type == "convnext":
            self.neck_cls = FPNConvnext
            self.head_cls = FPNHeadConvnext
        else:
            raise ValueError(f"fpn_type: {fpn_type} not supported")

        # Define the FPN neck
        self.neck = self.neck_cls(
            in_channels=self.in_channels,  # Input channels for FPN neck
            out_channels=self.out_channel,  # Number of channels in FPN output
            num_outs=self.num_outs,  # Number of FPN output scales
        )

        # Define the FPN head
        self.head = self.head_cls(
            in_channels=[
                self.out_channel for _ in range(self.num_outs)
            ],  # Input channels for FPN head
            in_index=self.in_index,  # Indices of the input feature maps
            feature_strides=self.feature_strides,  # Strides of the input feature maps
            channels=self.head_last_channel,  # Number of channels in intermediate layers
            dropout_ratio=dropout_ratio,  # Dropout ratio
            num_classes=n_classes,  # Number of output classes
            norm_cfg=dict(type="BN", requires_grad=True),  # Batch normalization config
            align_corners=False,  # Align corners for upsampling
        )

    def forward(self, x):
        # Forward pass through the backbone to get feature maps
        backbone_outputs = self.backbone(x)

        # Pass the feature maps from the backbone through the FPN neck
        # we only want the last num_outs elements
        fpn_outputs = self.neck(backbone_outputs[-self.num_outs :])

        # Pass the FPN outputs through the FPN head to get the final segmentation map
        output = self.head(fpn_outputs)

        return output


class MobileNetV2FPN(torch.nn.Module):
    """
    implement MobileNetV2 with FPN using transformers api.
    Better to use the OCDFPN class instead.
    """

    def __init__(
        self,
        width_multiplier=0.75,
        n_classes=1,
        num_outs=4,
        out_channel=32,  # output channel of FPN neck
        dropout_ratio=0,
        fpn_type="mm",
    ):
        super(MobileNetV2FPN, self).__init__()

        configuration = MobileNetV2Config(depth_multiplier=width_multiplier)
        self.backbone = MobileNetV2Model(configuration)
        self.stage_indices = [1, 4, 11, -1]

        if width_multiplier == 0.75:
            in_channels = [24, 24, 72, 240]
        elif width_multiplier == 1.0:
            in_channels = [24, 32, 96, 320]
        elif width_multiplier == 0.5:
            in_channels = [16, 16, 48, 160]
        else:
            raise ValueError("width_multiplier not supported")

        # only feed last num_outs channels to FPN
        self.out_channel = out_channel
        self.num_outs = num_outs
        self.in_channels = in_channels[-self.num_outs :]
        self.out_channels = [self.out_channel for _ in range(self.num_outs)]
        if n_classes < 24:
            self.head_last_channel = in_channels[0]
        else:
            self.head_last_channel = 80
        # backbone_outputs always has 5 elements, but we only want the last num_outs elements
        self.feature_strides = [4, 8, 16, 32][-self.num_outs :]
        self.in_index = [0, 1, 2, 3][: self.num_outs]

        # define fpn internal module type
        if fpn_type == "mm":
            self.neck_cls = FPN
            self.head_cls = FPNHead
        elif fpn_type == "extra_dw":
            self.neck_cls = FPNExtraDW
            self.head_cls = FPNHeadExtraDW
        elif fpn_type == "ib":
            self.neck_cls = FPNIB
            self.head_cls = FPNHeadIB
        elif fpn_type == "convnext":
            self.neck_cls = FPNConvnext
            self.head_cls = FPNHeadConvnext
        else:
            raise ValueError(f"fpn_type: {fpn_type} not supported")

        # Define the FPN neck
        self.neck = self.neck_cls(
            in_channels=self.in_channels,  # Input channels for FPN neck
            out_channels=self.out_channel,  # Number of channels in FPN output
            num_outs=self.num_outs,  # Number of FPN output scales
        )

        # Define the FPN head
        self.head = self.head_cls(
            in_channels=[
                self.out_channel for _ in range(self.num_outs)
            ],  # Input channels for FPN head
            in_index=self.in_index,  # Indices of the input feature maps
            feature_strides=self.feature_strides,  # Strides of the input feature maps
            channels=self.head_last_channel,  # Number of channels in intermediate layers
            dropout_ratio=dropout_ratio,  # Dropout ratio
            num_classes=n_classes,  # Number of output classes
            norm_cfg=dict(type="BN", requires_grad=True),  # Batch normalization config
            align_corners=False,  # Align corners for upsampling
        )

    def forward(self, x):
        # Forward pass through the backbone to get feature maps
        outputs = self.backbone(x, output_hidden_states=True)["hidden_states"]
        backbone_outputs = [outputs[i] for i in self.stage_indices]

        # Pass the feature maps from the backbone through the FPN neck
        # we only want the last num_outs elements
        fpn_outputs = self.neck(backbone_outputs[-self.num_outs :])

        # Pass the FPN outputs through the FPN head to get the final segmentation map
        output = self.head(fpn_outputs)

        return output


class OCDPAFPN(OCDFPN):
    def __init__(
        self,
        backbone="mobilenetv4_conv_medium.e500_r256_in1k",
        n_classes=1,
        num_outs=4,
        out_channel=32,  # output channel of FPN
        dropout_ratio=0.1,  #
        fpn_type="mm",
    ):
        super(OCDPAFPN, self).__init__(
            backbone, n_classes, num_outs, out_channel, dropout_ratio
        )
        if fpn_type == "mm":
            self.neck_cls = PAFPN
        elif fpn_type == "extra_dw":
            self.neck_cls = PAFPNExtraDW
        elif fpn_type == "ib":
            self.neck_cls = PAFPNIB
        elif fpn_type == "convnext":
            self.neck_cls = PAFPNConvnext
        else:
            raise ValueError(f"fpn_type: {fpn_type} not supported")

        # Define the PAFPN neck
        self.neck = self.neck_cls(
            in_channels=self.in_channels,  # Input channels for FPN neck
            out_channels=self.out_channel,  # Number of channels in FPN output
            num_outs=self.num_outs,  # Number of FPN output scales
        )


class FPNExtraDW(FPN):
    def __init__(self, *args, **kwargs):
        super(FPNExtraDW, self).__init__(*args, **kwargs)
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,  # changed from 3 to 2, might not compatabile with old models
                stride=1,
                start_dw_kernel_size=3,
                middle_dw_kernel_size=3,  # changed from 5 to 3
                activation="relu",
            )
            self.fpn_convs.append(fpn_conv)


class FPNIB(FPN):
    def __init__(self, *args, **kwargs):
        super(FPNIB, self).__init__(*args, **kwargs)
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,  # changed from 3 to 2, might not compatabile with old models
                stride=1,
                start_dw_kernel_size=0,
                middle_dw_kernel_size=3,
                activation="relu",
            )
            self.fpn_convs.append(fpn_conv)


class FPNConvnext(FPN):
    def __init__(self, *args, **kwargs):
        super(FPNConvnext, self).__init__(*args, **kwargs)
        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,
                stride=1,
                start_dw_kernel_size=3,
                middle_dw_kernel_size=0,
                activation="relu",
            )
            self.fpn_convs.append(fpn_conv)


class FPNHeadExtraDW(FPNHead):

    def __init__(self, *args, **kwargs):
        super(FPNHeadExtraDW, self).__init__(*args, **kwargs)

        self.scale_heads = nn.ModuleList()
        for i in range(len(self.feature_strides)):
            head_length = max(
                1,
                int(
                    np.log2(self.feature_strides[i]) - np.log2(self.feature_strides[0])
                ),
            )
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    UniversalInvertedBottleneckBlock(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        expand_ratio=2.0,  # changed from 3 to 2, might not compatabile with old models,
                        stride=1,
                        start_dw_kernel_size=3,
                        middle_dw_kernel_size=3,  # changed from 5 to 3, might not compatabile with old models
                        activation="relu",
                    )
                )
                if self.feature_strides[i] != self.feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=self.align_corners,
                        )
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))


class FPNHeadIB(FPNHead):

    def __init__(self, *args, **kwargs):
        super(FPNHeadIB, self).__init__(*args, **kwargs)

        self.scale_heads = nn.ModuleList()
        for i in range(len(self.feature_strides)):
            head_length = max(
                1,
                int(
                    np.log2(self.feature_strides[i]) - np.log2(self.feature_strides[0])
                ),
            )
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    UniversalInvertedBottleneckBlock(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        expand_ratio=2.0,  # changed from 3 to 2, might not compatabile with old models,
                        stride=1,
                        start_dw_kernel_size=0,
                        middle_dw_kernel_size=3,
                        activation="relu",
                    )
                )
                if self.feature_strides[i] != self.feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=self.align_corners,
                        )
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))


class FPNHeadConvnext(FPNHead):
    def __init__(self, *args, **kwargs):
        super(FPNHeadConvnext, self).__init__(*args, **kwargs)

        self.scale_heads = nn.ModuleList()
        for i in range(len(self.feature_strides)):
            head_length = max(
                1,
                int(
                    np.log2(self.feature_strides[i]) - np.log2(self.feature_strides[0])
                ),
            )
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    UniversalInvertedBottleneckBlock(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        expand_ratio=2.0,
                        stride=1,
                        start_dw_kernel_size=3,
                        middle_dw_kernel_size=0,
                        middle_dw_downsample=False,
                        activation="relu",
                    )
                )
                if self.feature_strides[i] != self.feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=self.align_corners,
                        )
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))


class PAFPNExtraDW(PAFPN):
    def __init__(self, *args, **kwargs):
        super(PAFPNExtraDW, self).__init__(*args, **kwargs)
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            d_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=0.0,
                stride=2,
                start_dw_kernel_size=3,
                middle_dw_kernel_size=3,  # changed from 5 to 3, might not compatabile with old models
                activation="relu",
            )
            pafpn_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,  # changed from 3 to 2, might not compatabile with old models,
                stride=1,
                start_dw_kernel_size=3,
                middle_dw_kernel_size=3,  # changed from 5 to 3, might not compatabile with old models
                activation="relu",
            )
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)


class PAFPNIB(PAFPN):
    def __init__(self, *args, **kwargs):
        super(PAFPNIB, self).__init__(*args, **kwargs)
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            d_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,  # changed from 3 to 2, might not compatabile with old models,
                stride=2,
                start_dw_kernel_size=0,
                middle_dw_kernel_size=3,
                activation="relu",
            )
            pafpn_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,  # changed from 3 to 2, might not compatabile with old models,
                stride=1,
                start_dw_kernel_size=0,
                middle_dw_kernel_size=3,
                activation="relu",
            )
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)


class PAFPNConvnext(PAFPN):
    def __init__(self, *args, **kwargs):
        super(PAFPNConvnext, self).__init__(*args, **kwargs)
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            d_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,
                stride=2,
                start_dw_kernel_size=3,
                middle_dw_kernel_size=0,
                middle_dw_downsample=False,
                activation="relu",
            )
            pafpn_conv = UniversalInvertedBottleneckBlock(
                self.out_channels,
                self.out_channels,
                expand_ratio=2.0,
                stride=1,
                start_dw_kernel_size=3,
                middle_dw_kernel_size=0,
                middle_dw_downsample=False,
                activation="relu",
            )
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)


if __name__ == "__main__":
    # Test the FPN model
    model = OCDFPN(
        backbone="mobilenetv4_conv_large.e500_r256_in1k",
        n_classes=1,
        num_outs=4,
        out_channel=128,
        dropout_ratio=0.1,
        fpn_type="mm",
    )
    x = torch.randn(1, 3, 320, 320)
    output = model(x)
    print(output.shape)
