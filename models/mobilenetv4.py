import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from utils.utils import make_divisible


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio,
        stride,
        middle_dw_downsample=True,
        start_dw_kernel_size=0,
        middle_dw_kernel_size=3,
        end_dw_kernel_size=0,
        activation="relu",
        depthwise_activation=None,
        # dilation=1,
        divisible_by=8,
        use_residual=True,
        use_layer_scale=False,
        layer_scale_init_value=1e-5,
    ):
        super(UniversalInvertedBottleneckBlock, self).__init__()
        self.stride = stride
        self.use_residual = use_residual and in_channels == out_channels and stride == 1
        self.use_layer_scale = use_layer_scale

        if depthwise_activation is None:
            depthwise_activation = activation

        expand_channels = make_divisible(in_channels * expand_ratio, divisible_by)

        # Starting depthwise conv
        if start_dw_kernel_size > 0:
            self.start_dw_conv = Conv2dNormActivation(
                in_channels,
                in_channels,
                kernel_size=start_dw_kernel_size,
                stride=stride if not middle_dw_downsample else 1,
                padding=start_dw_kernel_size // 2,
                groups=in_channels,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,  # No activation after start_dw_conv
            )
        else:
            self.start_dw_conv = None

        # Expansion with 1x1 conv
        self.expand_conv = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=expand_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU if activation == "relu" else nn.ReLU6,
            inplace=True,
        )

        # Middle depthwise conv
        if middle_dw_kernel_size > 0:
            self.middle_dw_conv = Conv2dNormActivation(
                in_channels=expand_channels,
                out_channels=expand_channels,
                kernel_size=middle_dw_kernel_size,
                stride=stride if middle_dw_downsample else 1,
                padding=middle_dw_kernel_size // 2,
                groups=expand_channels,
                norm_layer=nn.BatchNorm2d,
                activation_layer=(
                    nn.ReLU if depthwise_activation == "relu" else nn.ReLU6
                ),
                inplace=True,
            )
        else:
            self.middle_dw_conv = None

        # Projection with 1x1 conv
        self.project_conv = Conv2dNormActivation(
            in_channels=expand_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,  # No activation after project_conv
        )

        # Ending depthwise conv
        if end_dw_kernel_size > 0:
            self.end_dw_conv = Conv2dNormActivation(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=end_dw_kernel_size,
                stride=1,
                padding=end_dw_kernel_size // 2,
                groups=out_channels,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,  # No activation after end_dw_conv
            )
        else:
            self.end_dw_conv = None

        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones(out_channels), requires_grad=True
            )
        else:
            self.layer_scale = None

    def forward(self, x):
        identity = x

        if self.start_dw_conv is not None:
            x = self.start_dw_conv(x)
            # No activation

        x = self.expand_conv(x)

        if self.middle_dw_conv is not None:
            x = self.middle_dw_conv(x)

        x = self.project_conv(x)

        if self.end_dw_conv is not None:
            x = self.end_dw_conv(x)
            # No activation

        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)

        if self.use_residual:
            x = x + identity

        return x


class MobileNetV4(nn.Module):
    def __init__(self, arch="small", n_classes=1000, width_multiplier=1.0, dropout=0.2):
        super(MobileNetV4, self).__init__()
        self.arch = arch
        self.n_classes = n_classes
        self.width_multiplier = width_multiplier
        self.dropout = dropout

        # Define architecture specs
        if arch == "small":
            block_specs = self._mnv4_conv_small_block_specs()
        elif arch == "medium":
            block_specs = self._mnv4_conv_medium_block_specs()
        elif arch == "medium_seg":
            block_specs = self._mnv4_conv_medium_seg_block_specs()
        elif arch == "large":
            block_specs = self._mnv4_conv_large_block_specs()
        else:
            raise ValueError(f"Unknown architecture {arch}")

        # Build the model in stages
        self.stages = nn.ModuleList()
        current_stage = None
        stage_layers = []
        in_channels = 3

        for spec in block_specs:
            stage = spec.get("stage", 1)  # Default to stage 1 if not specified
            if current_stage is None:
                current_stage = stage
            elif stage != current_stage:
                # Append the previous stage layers
                self.stages.append(nn.Sequential(*stage_layers))
                stage_layers = []
                current_stage = stage

            block_fn = spec["block_fn"]
            if block_fn == "convbn":
                out_channels = make_divisible(
                    spec["out_channels"] * self.width_multiplier, 8
                )
                activation_layer = (
                    nn.ReLU if spec.get("activation", "relu") == "relu" else nn.ReLU6
                )
                layer = Conv2dNormActivation(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=spec["kernel_size"],
                    stride=spec["strides"],
                    activation_layer=activation_layer,
                    inplace=True,
                )
                stage_layers.append(layer)
                in_channels = out_channels
            elif block_fn == "uib":
                out_channels = make_divisible(
                    spec["out_channels"] * self.width_multiplier, 8
                )
                expand_ratio = spec["expand_ratio"]
                stride = spec["strides"]
                start_dw_kernel_size = spec.get("start_dw_kernel_size", 0)
                middle_dw_kernel_size = spec.get("middle_dw_kernel_size", 0)
                end_dw_kernel_size = spec.get("end_dw_kernel_size", 0)
                middle_dw_downsample = spec.get("middle_dw_downsample", True)
                use_layer_scale = spec.get("use_layer_scale", False)
                layer = UniversalInvertedBottleneckBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expand_ratio=expand_ratio,
                    stride=stride,
                    middle_dw_downsample=middle_dw_downsample,
                    start_dw_kernel_size=start_dw_kernel_size,
                    middle_dw_kernel_size=middle_dw_kernel_size,
                    end_dw_kernel_size=end_dw_kernel_size,
                    activation=spec.get("activation", "relu"),
                    use_layer_scale=use_layer_scale,
                )
                stage_layers.append(layer)
                in_channels = out_channels
            # Skipping 'gpooling' and the last 'convbn' as per your request

        # Append the last stage
        if stage_layers:
            self.stages.append(nn.Sequential(*stage_layers))

        if self.n_classes > 0:

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            last_channels = in_channels
            self.classifier = nn.Sequential(
                nn.Conv2d(
                    last_channels, 1280, kernel_size=1, stride=1, padding=0, bias=True
                ),
                nn.ReLU6(inplace=True),
                nn.Dropout(p=self.dropout),
                nn.Flatten(),
                nn.Linear(1280, self.n_classes),
            )

    def forward(self, x):
        # Use forward_features to get the last feature map
        x = self.forward_features(x)
        if self.n_classes > 0:
            x = self.avgpool(x[-1])
            x = self.classifier(x)
        return x

    def forward_features(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def _mnv4_conv_small_block_specs(self):
        # Define the block specs for the small architecture
        block_specs = [
            # stage 1, 112x112 (dim after the following layer)
            {
                "stage": 1,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 32,
            },  # conv
            # stage 2, 56x56, use 2 conv layers to represent FusedIB with stride 2
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 32,
            },
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 32,
            },
            # stage 3, 28x28, , use 2 conv layers to represent FusedIB with stride 2
            {
                "stage": 3,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 96,
            },
            {
                "stage": 3,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 64,
            },
            # stage 4, 14x14
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 96,
                "expand_ratio": 3.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 5,
            },  # ExtraDW
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 96,
                "expand_ratio": 2.0,
                "middle_dw_kernel_size": 3,
            },  # IB
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 96,
                "expand_ratio": 2.0,
                "middle_dw_kernel_size": 3,
            },  # IB
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 96,
                "expand_ratio": 2.0,
                "middle_dw_kernel_size": 3,
            },  # IB
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 96,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
            },  # ConvNext
            # stage 5, 7x7
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 128,
                "expand_ratio": 6.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },  # ExtraDW
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 5,
            },  # ExtraDW
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "middle_dw_kernel_size": 5,
            },  # IB
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 3.0,
                "middle_dw_kernel_size": 5,
            },  # IB
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "middle_dw_kernel_size": 3,
            },  # IB
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "middle_dw_kernel_size": 3,
            },  # IB
            {
                "stage": 5,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 960,
            },  # conv
            # {'block_fn': 'gpooling'},
            # {'block_fn': 'convbn', 'activation': 'relu', 'kernel_size': 1, 'strides': 1, 'out_channels': 1280},
        ]
        return block_specs

    def _mnv4_conv_medium_block_specs(self):
        # Define the block specs for the medium architecture
        block_specs = [
            # Stage 1
            {
                "stage": 1,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 32,
            },
            # Stage 2
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 32,
            },
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 32,
            },
            # Stage 3
            {
                "stage": 3,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 80,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 3,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 80,
                "expand_ratio": 2.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },
            # Stage 4
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 160,
                "expand_ratio": 6.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },
            # Repeating blocks in Stage 4
            *[
                {
                    "stage": 4,
                    "block_fn": "uib",
                    "activation": "relu",
                    "strides": 1,
                    "out_channels": 160,
                    "expand_ratio": 4.0,
                    "start_dw_kernel_size": 3 if i % 2 == 0 else 0,
                    "middle_dw_kernel_size": 5 if i == 2 else (3 if i % 2 == 0 else 0),
                }
                for i in range(4)
            ],
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 2.0,
                "start_dw_kernel_size": 0,
                "middle_dw_kernel_size": 0,
            },
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 0,
            },
            # Stage 5
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 256,
                "expand_ratio": 6.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 5,
            },
            *[
                {
                    "stage": 5,
                    "block_fn": "uib",
                    "activation": "relu",
                    "strides": 1,
                    "out_channels": 256,
                    "expand_ratio": 4.0,
                    "start_dw_kernel_size": ks[0],
                    "middle_dw_kernel_size": ks[1],
                }
                for ks in [(5, 5), (3, 5), (3, 5), (0, 0), (3, 0), (3, 5), (5, 5)]
            ],
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 256,
                "expand_ratio": 2.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 0,
            },
            # FC layers
            {
                "stage": 5,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 960,
            },
            # {"block_fn": "gpooling"},
            # {
            #     "block_fn": "convbn",
            #     "activation": "relu",
            #     "kernel_size": 1,
            #     "strides": 1,
            #     "out_channels": 1280,
            # },
        ]
        return block_specs

    def _mnv4_conv_medium_seg_block_specs(self):
        # Define the block specs for the medium segmentation architecture
        # Tailored MobileNetV4ConvMedium for dense prediction, e.g. segmentation
        block_specs = [
            # Stage 1
            {
                "stage": 1,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 32,
            },
            # Stage 2
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 32,
            },
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 32,
            },
            # Stage 3
            {
                "stage": 3,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 80,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 3,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 80,
                "expand_ratio": 2.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },
            # Stage 4
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 160,
                "expand_ratio": 6.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            # Repeating blocks in Stage 4
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 0,
            },
            # This block is marked as output in the original TensorFlow code
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 160,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 0,
            },
            # Stage 5
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 256,
                "expand_ratio": 6.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 0,
            },
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 2.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 128,
                "expand_ratio": 2.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 0,
            },
            # FC layers
            {
                "stage": 5,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 448,
            },
            # {"block_fn": "gpooling"},
            # {
            #     "stage": 5,
            #     "block_fn": "convbn",
            #     "activation": "relu",
            #     "kernel_size": 1,
            #     "strides": 1,
            #     "out_channels": 1280,
            # },
        ]
        return block_specs

    def _mnv4_conv_large_block_specs(self):
        # Define the block specs for the large architecture
        block_specs = [
            # Stage 1
            {
                "stage": 1,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 24,
            },
            # Stage 2
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 3,
                "strides": 2,
                "out_channels": 32,
            },
            {
                "stage": 2,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 32,
            },
            # Stage 3
            {
                "stage": 3,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 96,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            {
                "stage": 3,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 96,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 3,
            },
            # Stage 4
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 192,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 5,
            },
            # Repeating blocks in Stage 4
            *[
                {
                    "stage": 4,
                    "block_fn": "uib",
                    "activation": "relu",
                    "strides": 1,
                    "out_channels": 192,
                    "expand_ratio": 4.0,
                    "start_dw_kernel_size": 3 if i % 2 == 0 else 5,
                    "middle_dw_kernel_size": 3 if i % 2 == 0 else 3,
                }
                for i in range(10)
            ],
            {
                "stage": 4,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 192,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 3,
                "middle_dw_kernel_size": 0,
            },
            # Stage 5
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 2,
                "out_channels": 512,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 5,
            },
            # Repeating blocks in Stage 5
            *[
                {
                    "stage": 5,
                    "block_fn": "uib",
                    "activation": "relu",
                    "strides": 1,
                    "out_channels": 512,
                    "expand_ratio": 4.0,
                    "start_dw_kernel_size": 5,
                    "middle_dw_kernel_size": ks,
                }
                for ks in [5, 5, 5, 5, 0, 3, 0, 0, 3, 5, 0, 0]
            ],
            {
                "stage": 5,
                "block_fn": "uib",
                "activation": "relu",
                "strides": 1,
                "out_channels": 512,
                "expand_ratio": 4.0,
                "start_dw_kernel_size": 5,
                "middle_dw_kernel_size": 0,
            },
            # FC layers
            {
                "stage": 5,
                "block_fn": "convbn",
                "activation": "relu",
                "kernel_size": 1,
                "strides": 1,
                "out_channels": 960,
            },
            # {"block_fn": "gpooling"},
            # {
            #     "block_fn": "convbn",
            #     "activation": "relu",
            #     "kernel_size": 1,
            #     "strides": 1,
            #     "out_channels": 1280,
            # },
        ]
        return block_specs
