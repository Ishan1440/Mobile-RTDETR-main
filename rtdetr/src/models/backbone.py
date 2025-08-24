from torch import nn, Tensor
from torch.nn.modules import padding

import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torch.jit.annotations import List
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from src.core import register
import torch.nn.functional as F
import torch
from torchinfo import summary

__all__ = ['MobileNetWithExtraBlocks']


class BackboneBase(nn.Module):
    # helps extract backbone features
    def __init__(
        self,
        backbone: nn.Module, # The CNN feature extractor (eg: pretrained MobileNet)
        # extra_blocks: nn.Module,
        train_backbone: bool, # whether to fine-tune the backbone or freeze it
        return_layers_backbone: dict,
        # return_layers_extra_blocks: dict,
    ):
        super().__init__()
        print("train_backbone: ", train_backbone)
        for _, parameter in backbone.named_parameters():
            if train_backbone:
                parameter.requires_grad=True
            else:
                parameter.requires_grad = False
        # Sets requires_grad flag for all backbone parameters. If False, gradients won’t be computed → backbone is frozen (useful when you only want to train new layers on top).

        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers_backbone)
            # to return the outputs (feature maps) of specified layers

        # self.temp_layer = nn.Sequential(
        #     torch.nn.Conv2d(40, 128, kernel_size=3,padding=1),
        #     torch.nn.Conv2d(112, 128, kernel_size=3,padding=1))
        # Would normalize output channels to 128 (for detection heads).

        # print(self.body)
        # self.extra_blocks = IntermediateLayerGetter(
        #     extra_blocks, return_layers=return_layers_extra_blocks)
        # to add extra convolutional layers after the backbone (like SSD)

    def forward(self, x: Tensor):
        xs_body = self.body(x) # dict of feature maps
        # make output channels = 128
        out: List[Tensor] = []
        i = 0

        for name, x in xs_body.items():
            # x=self.temp_layer[i](x)
            out.append(x) 
            i += 1

        # xs_extra_blocks = self.extra_blocks(out[-1])
        # for name, x in xs_extra_blocks.items():
        #     out.append(x)
        return out


@register
class MobileNetWithExtraBlocks(BackboneBase):
    """MobileNet backbone with extra blocks (optional).
    Builds a MobileNetV3 backbone (large/small). Selects feature maps from intermediate layers for multi-scale use."""

    def __init__(
        self,
        train_backbone: bool=True,
        backbone_size: str = "large", # to choose between "large" (MobileNetV3-Large) and "small" (MobileNetV3-Small).
        load_pretrained:bool=True, # whether to initialize MobileNet with pretrained ImageNet weights.

    ):
        if backbone_size == "large":
            print("Pretrained: ",load_pretrained)
            backbone = torchvision.models.mobilenet_v3_large(
                pretrained=load_pretrained).features #only .features is loaded that is the convolutional layers, not the classifier
            return_layers_backbone = {
                "6": "0", # layer 6 -> feature map 0
                "12": "1",
                "14": "2",
            }
        elif backbone_size == "small":
            backbone = torchvision.models.mobilenet_v3_small(
                pretrained=load_pretrained).features
            return_layers_backbone = {
                "3": "0",
                "8": "1",
                "11": "2",
            }
        # for name, m in backbone.named_children():
        #     print(name)
        #
        summary(backbone, input_size=(4, 3, 800, 800), depth=1)
        # Prints a summary of MobileNet backbone layers for a batch size of 4 and image size 800x800

        num_channels = 128 # input channels from last backbone stage
        hidden_dims = [256, 512] # output sizes of extra blocks
        expand_ratios = [0.25] # how much to expand intermediate channels in inverted residuals?
        strides = [2, 1, 2] # controls downsampling
        # extra_blocks = ExtraBlocks(
        #     num_channels, hidden_dims, expand_ratios, strides)
        # return_layers_extra_blocks = {"0": "2", }
        # These define how the extra layers (ExtraBlocks) would be constructed if enabled. To add extra inverted residual layers for more feature maps

        super().__init__(
            backbone,
            # extra_blocks,
            train_backbone,
            return_layers_backbone,
            # return_layers_extra_blocks,
        )


class ExtraBlocks(nn.Sequential):
    # defines additional layers on top of the backbone 

    def __init__(self, in_channels, hidden_dims, expand_ratios, strides):
        extra_blocks = []
        for i in range(len(expand_ratios)):
            input_dim = hidden_dims[i - 1] if i > 0 else in_channels

            # inverted residual layers
            extra_blocks.append(
                InvertedResidual(
                    InvertedResidualConfig(
                        input_channels=input_dim,
                        kernel=3, 
                        expanded_channels=input_dim*expand_ratios[i],
                        out_channels=hidden_dims[i],
                        use_se=False, # No squeeze and excitation
                        activation="RE",  # ReLU activation
                        stride=strides[i],
                        dilation=1,
                        width_mult=1.0,
                        # applied depth-wise convolution
                    ),
                    norm_layer=nn.BatchNorm2d,
                )
            )

        super().__init__(*extra_blocks)
        # makes ExtraBlocks just a nn.Sequential container of inverted residual blocks

