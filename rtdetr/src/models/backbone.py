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

    def __init__(
        self,
        backbone: nn.Module,
        # extra_blocks: nn.Module,
        train_backbone: bool,
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

        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers_backbone)
        # self.temp_layer = nn.Sequential(
        #     torch.nn.Conv2d(40, 128, kernel_size=3,padding=1),
        #     torch.nn.Conv2d(112, 128, kernel_size=3,padding=1))
        # print(self.body)
        # self.extra_blocks = IntermediateLayerGetter(
        #     extra_blocks, return_layers=return_layers_extra_blocks)

    def forward(self, x: Tensor):
        xs_body = self.body(x)
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
    """MobileNet backbone with extra blocks."""

    def __init__(
        self,
        train_backbone: bool=True,
        backbone_size: str = "large",
        load_pretrained:bool=True,

    ):
        if backbone_size == "large":
            print("Pretrained: ",load_pretrained)
            backbone = torchvision.models.mobilenet_v3_large(
                pretrained=load_pretrained).features
            return_layers_backbone = {
                "6": "0",
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

        num_channels = 128
        hidden_dims = [256, 512]
        expand_ratios = [0.25]
        strides = [2, 1, 2]
        # extra_blocks = ExtraBlocks(
        #     num_channels, hidden_dims, expand_ratios, strides)
        # return_layers_extra_blocks = {"0": "2", }

        super().__init__(
            backbone,
            # extra_blocks,
            train_backbone,
            return_layers_backbone,
            # return_layers_extra_blocks,
        )


class ExtraBlocks(nn.Sequential):
    def __init__(self, in_channels, hidden_dims, expand_ratios, strides):
        extra_blocks = []
        for i in range(len(expand_ratios)):
            input_dim = hidden_dims[i - 1] if i > 0 else in_channels
            extra_blocks.append(
                InvertedResidual(
                    InvertedResidualConfig(
                        input_channels=input_dim,
                        kernel=3,
                        expanded_channels=input_dim*expand_ratios[i],
                        out_channels=hidden_dims[i],
                        use_se=False,
                        activation="RE",
                        stride=strides[i],
                        dilation=1,
                        width_mult=1.0,

                    ),
                    norm_layer=nn.BatchNorm2d,
                )
            )

        super().__init__(*extra_blocks)

