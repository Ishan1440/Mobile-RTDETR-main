# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn


class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(B, C, P, N)` where :math:`B` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        # print("Using Linear Attention")

        self.qkv_proj = nn.Linear(
            in_features=embed_dim,
            out_features=1 + (2 * embed_dim),
            bias=bias,
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    def _forward_self_attn(self, x, *args, **kwargs) -> tuple[Tensor, None]:
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, N]
        # value, key --> [B, d, N]
        query, key, value = torch.split(
            qkv,
            split_size_or_sections=[1, self.embed_dim, self.embed_dim],
            dim=-1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, N] x [B, 1, N] -> [B, d, N]
        context_vector = key * context_scores
        # [B, d, N] --> [B, d, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, N] * [B, d, 1] --> [B, d, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return (out, None)

    def forward(
        self, x, *args, **kwargs
    ) -> tuple[Tensor, None]:
        return self._forward_self_attn(x, *args, **kwargs)


class ConvertToX(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.w = nn.Parameter(torch.rand(size=(3,)), requires_grad=True)

    def forward(self, q, k, v) -> torch.Tensor:
        # (b,s,e) -> (b,1,s,e)
        x = torch.cat([q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)], dim=1)
        x = self.w.view(-1, 1, 1) * x
        return x.sum(dim=1)
