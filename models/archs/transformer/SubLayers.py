''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from models.archs.transformer.Modules import spaBlock,freBlock

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Attention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, dim, num_heads, bias,dropout=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))
        self.dropout = nn.Dropout(dropout)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
    def forward(self, x , mask = None):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v_pos = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_pos, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn0 = (q @ k.transpose(-2, -1)) * self.temperature

        if mask is not None:
            attn1 = attn0.masked_fill(mask == 0, -1e9)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))


        attn0 = self.softmax(attn0)
        attn1 = self.softmax(attn1)
        attn = attn0 * w1 + attn1 * w2

        attn = self.dropout(attn)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)    #b c h w
        out_p = self.pos_emb(v_pos)
        out = out_p + out
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2, bias = False):

        super(FeedForward, self).__init__()


        self.dim = dim
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.fre = freBlock(hidden_features, bias)
        self.spa = spaBlock(hidden_features, bias)

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        x = self.project_in(x)
        _,_,h,w = x.shape
        x_spa = self.spa(x)
        x_fre = self.fre(x)
        x = torch.cat([x_spa, x_fre], dim = 1)
        x = self.project_out(x)
        return x
