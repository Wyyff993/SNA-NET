''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from models.archs.transformer.Layers import EncoderLayer3
import torch.nn.functional as F


class Encoder_patch66(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, dim, heads, ffn_expansion_factor, num_blocks, bias = False, LayerNorm_type = "WithBias"):
        super(Encoder_patch66, self).__init__()

        self.layer = EncoderLayer3(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type)
        self.blocks = num_blocks

    def forward(self, x, mask):

        for i in range(self.blocks):
            out = self.layer(x, mask = mask)
        return out

