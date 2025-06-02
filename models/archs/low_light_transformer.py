import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.arch_util import feature_fusion
import numpy as np
import cv2

from models.archs.arch_util import Down
from models.archs.transformer.Models import Encoder_patch66

###############################
class low_light_transformer(nn.Module):
    def __init__(self, nf=48, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True,
                 num_blocks = [4,6,6,8],
                 heads = [1,2,4,8],
                 ffn_expansion_factor = 2):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf * 8)


        #卷积
        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf // 2, 3, 1, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv_first_4 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)

        self.upconv1 = nn.Conv2d(nf * 8, nf * 16, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=True)
        self.reduce_1 = nn.Conv2d(nf * 8, nf * 4, kernel_size=1, bias=False)
        self.upconv3 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.reduce_2 = nn.Conv2d(nf * 4, nf * 2, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.reduce_3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        #transforemr
        self.encoder_level1 = Encoder_patch66(dim=nf, heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                              num_blocks=num_blocks[0])
        self.down1_2 = Down(nf)
        self.encoder_level2 = Encoder_patch66(dim=nf * 2 ** 1, heads=heads[1],
                                              ffn_expansion_factor=ffn_expansion_factor,
                                              num_blocks=num_blocks[1])
        self.down2_3 = Down(nf* 2 ** 1)
        self.encoder_level3 = Encoder_patch66(dim=nf * 2 ** 2, heads=heads[2],
                                              ffn_expansion_factor=ffn_expansion_factor,
                                              num_blocks=num_blocks[2])
        self.down3_4 = Down(nf * 2 ** 2)
        self.encoder_level4 = Encoder_patch66(dim=nf * 2 ** 3, heads=heads[3],
                                              ffn_expansion_factor=ffn_expansion_factor,
                                              num_blocks=num_blocks[3])

        #self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        #self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)
        self.fusion_1 = feature_fusion(feat= nf)
        self.fusion_2 = feature_fusion(feat= nf * 2)
        self.fusion_3 = feature_fusion(feat= nf * 4)
        self.fusion_4 = feature_fusion(feat= nf * 8)


    def forward(self, x, mask=None):
        x_center = x
        t = 0.5  #0.1
        mask_t = F.interpolate(mask, size=[self.nf, self.nf], mode='nearest')
        mask_t[mask_t <= t] = 0.0

        # level 1
        fea = self.conv_first_1(x_center)
        L1_fea_1 = self.lrelu(fea)
        # 全局
        fea_trans1 = self.encoder_level1(fea, mask_t)
        fea_1 = self.fusion_1(L1_fea_1, fea_trans1, mask)

        #level 2
        L1_fea_2 = self.lrelu(self.pixel_unshuffle(self.conv_first_2(L1_fea_1)))
        #全局
        fea_trans2 = self.down1_2(fea_trans1)
        fea_trans2 = self.encoder_level2(fea_trans2, mask_t)
        fea_2 = self.fusion_2(L1_fea_2, fea_trans2, mask)

        #level 3
        L1_fea_3 = self.lrelu(self.pixel_unshuffle(self.conv_first_3(L1_fea_2)))
        #全局
        fea_trans3 = self.down2_3(fea_trans2)
        fea_trans3 = self.encoder_level3(fea_trans3, mask_t)
        fea_3 = self.fusion_3(L1_fea_3, fea_trans3, mask)

        #level4
        L1_fea_4 = self.lrelu(self.pixel_unshuffle(self.conv_first_4(L1_fea_3)))
        #全局
        fea_trans4 = self.down3_4(fea_trans3)
        fea_trans4 = self.encoder_level4(fea_trans4, mask_t)
        fea_4 = self.fusion_4(L1_fea_4, fea_trans4, mask)

        # bottle
        out = self.recon_trunk(fea_4)

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))   #4
        out = torch.cat([out, fea_3], dim=1)
        out = self.reduce_1(out)

        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))     #2
        out = torch.cat([out, fea_2], dim=1)
        out = self.reduce_2(out)

        out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))     #1
        out = torch.cat([out, fea_1], dim=1)
        out = self.lrelu(self.reduce_3(out))
        out = self.conv_last(out) + x_center

        return out
