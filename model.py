import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import numpy as np

class NormFeat(nn.Module):
    ''' L2 normalization of features '''
    def __init__(self, scale_factor=1.0):
        super(NormFeat, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return self.scale_factor * F.normalize(input, p=2, dim=1)

class Net(nn.Module):
    def __init__(self, backbone, num_class, alpha_1, feat_dim=128):
        super(Net, self).__init__()
        self.backbone = backbone
        self.num_class = num_class
        self.alpha_1 = alpha_1
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(1000, self.feat_dim)#nn.Linear(512, self.feat_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        self.batch_norm = nn.BatchNorm2d(self.feat_dim)
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()
        self.relu = nn.ReLU(inplace=True)
        self.norm_feat = NormFeat()
        self.fc2 = nn.Linear(self.feat_dim, self.num_class)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x).view(batch_size, -1)
#         print('backbone', x.shape)
        x = self.fc1(x).unsqueeze(dim=2).unsqueeze(dim=3)
        x = self.relu(self.batch_norm(x))
        feat = self.norm_feat(x).squeeze() * self.alpha_1
        out = self.fc2(feat)
        return feat, out

class SimpleNet(nn.Module):
    def __init__(self, backbone, num_class, feat_dim=512):
        super(SimpleNet, self).__init__()
        self.backbone = backbone
        self.num_class = num_class
        self.fc1 = nn.Linear(512, self.num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x).view(batch_size, -1) 
        feat = x.squeeze()
        out = self.fc1(feat) 
        return feat, out

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )
class MobNet(nn.Module):
    def __init__(self, n_class, feat_dim=512):
        super(MobNet, self).__init__()
        self.n_class = n_class
        self.feat_dim = feat_dim
        self.conv_bn_0 = conv_bn(1, 3, 2)
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    
            conv_dw(8, 16, 1),   
            conv_dw(16, 32, 2),  
            conv_dw(32, 32, 1),  
            conv_dw(32, 64, 2),  
            conv_dw(64, 64, 1),  
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), 
            conv_dw(256, self.feat_dim, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.feat_dim, self.n_class)

    def forward(self, x):
        x = self.conv_bn_0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat = self.avg(x).squeeze()
        # feat = x.view(-1, self.feat_dim)
        # print('feat: ', feat.shape)
        out = self.fc(feat)
        return feat, out
 