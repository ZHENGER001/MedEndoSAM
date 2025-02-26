import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def get_same_padding(kernel_size: int or Tuple[int, ...]) -> int or Tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


# Activation function construction
def build_act(name: str, **kwargs) -> nn.Module or None:
    act_dict = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    if name in act_dict:
        return act_dict[name](**kwargs)
    return None


# Convolutional Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, dilation=1, groups=1,
                 use_bias=False, dropout=0, norm="bn2d", act_func="relu"):
        super(ConvLayer, self).__init__()
        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride), padding=padding,
                              dilation=(dilation, dilation), groups=groups, bias=use_bias)
        self.norm = build_act(norm, num_features=out_channels) if norm else None
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# Weighted Multi-Scale Fusion Module
class WeightedMultiScaleMLA(nn.Module):
    """Weighted Multi-Scale Fusion Module"""

    def __init__(self, in_channels: int, out_channels: int, base_scales: Tuple[int, ...] = (3, 5, 7),
                 use_bias=False, norm="bn2d", act_func="relu"):
        super(WeightedMultiScaleMLA, self).__init__()

        self.base_scales = base_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build convolution layers based on different scales
        self.scales_conv = nn.ModuleList([
            ConvLayer(in_channels, out_channels, kernel_size=scale, use_bias=use_bias, norm=norm, act_func=act_func)
            for scale in self.base_scales
        ])

        # Learn a weight for each scale
        self.scale_weights = nn.Parameter(torch.ones(len(self.base_scales)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the output for each scale
        scale_outputs = [conv_layer(x) for conv_layer in self.scales_conv]

        # Apply weights to the output of each scale
        weighted_output = sum(w * scale_output for w, scale_output in zip(self.scale_weights, scale_outputs))
        return weighted_output


# Test the Weighted Multi-Scale Fusion Module
if __name__ == '__main__':
    block = WeightedMultiScaleMLA(in_channels=64, out_channels=64)
    input1 = torch.rand(3, 64, 32, 32)  # Input size of 32x32
    output = block(input1)
    print(input1.size())
    print(output.size())

    input2 = torch.rand(3, 64, 256, 256)  # Input size of 256x256
    output = block(input2)
    print(input2.size())
    print(output.size())
