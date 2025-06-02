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

