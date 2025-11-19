import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class Down(nn.Module):
    def __init__(self, feat=64):
        super(Down, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(feat, feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class feature_fusion(nn.Module):
    def __init__(self, feat=64, dropout=0.0):
        super(feature_fusion, self).__init__()
        d = feat // 2
        self.fc = nn.Linear(feat, d)
        self.fc_c = nn.Linear(d, feat)
        self.fc_t = nn.Linear(d, feat)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, c_fea, t_fea, mask):
        _, c, h, w = c_fea.shape

        mask = F.interpolate(mask, size=[h, w], mode='nearest')
        mask = mask.repeat(1, c, 1, 1)

        fea1 = (c_fea * mask).unsqueeze(dim=1)
        fea2 = (t_fea * 1 - mask).unsqueeze(dim=1)

        feas = torch.cat([fea1, fea2], dim=1)
        fea_u = torch.sum(feas, dim=1)
        fea_s = fea_u.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        fea_z = self.dropout(fea_z)

        vector_c = self.fc_c(fea_z).unsqueeze(dim=1)
        vector_t = self.fc_t(fea_z).unsqueeze(dim=1)
        vectors = self.softmax(torch.cat([vector_c, vector_t], dim=1)).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * vectors).sum(dim=1)
        return fea_v


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

