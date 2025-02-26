import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv3WithSpatialAttention(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()  # Keep the original input unchanged
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        # Add spatial attention
        attention_map = self.spatial_attention(x)
        x = x * attention_map

        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # Add spatial attention
        attention_map = self.spatial_attention(x)
        x = x * attention_map

        return x


if __name__ == '__main__':
    block = PartialConv3WithSpatialAttention(64, 2, 'split_cat').cuda()
    input_tensor = torch.rand(1, 64, 64, 64).cuda()
    output = block(input_tensor)
    print(input_tensor.size(), output.size())
