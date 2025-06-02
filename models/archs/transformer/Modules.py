import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class freBlock(nn.Module):
    def __init__(self, hidden_features, bias):
        super(freBlock, self).__init__()

        self.patch_size = 8
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.complex_weight = nn.Parameter(torch.randn(hidden_features * 2, 1, 1, 2, dtype=torch.float32) * 0.02)
    def forward(self, x):
        b,c,h,w = x.shape
        x = torch.fft.rfft2(x, dim=(2,3), norm='ortho')
        fft = torch.view_as_complex(self.complex_weight)
        x = x * fft
        x = torch.fft.irfft2(x, s=(h, w), dim=(2, 3), norm='ortho')

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2

        return x


class spaBlock(nn.Module):
    def __init__(self, hidden_features, bias):
        super(spaBlock, self).__init__()

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)


    def forward(self, x):
            x1, x2 = self.dwconv(x).chunk(2, dim=1)
            x = F.gelu(x1) * x2

            return x