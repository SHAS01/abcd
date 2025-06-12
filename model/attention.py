###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class ChannelAttention(Module):
    """Channel attention module for retinal vessel features"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(Module):
    """Spatial attention module for retinal vessel features"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        out = self.bn(out)
        return self.sigmoid(out)

class VesselAttention(Module):
    """Combined attention module specifically designed for vessel segmentation"""
    def __init__(self, in_channels):
        super(VesselAttention, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()
        
        # Additional vessel-specific attention
        self.vessel_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        channel_att = self.channel_att(x)
        x = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_att(x)
        x = x * spatial_att
        
        # Vessel-specific attention
        vessel_att = self.vessel_conv(x)
        x = x * vessel_att
        
        return x

class MultiScaleVesselAttention(Module):
    """Multi-scale attention module for capturing vessels at different scales"""
    def __init__(self, in_channels):
        super(MultiScaleVesselAttention, self).__init__()
        
        self.scales = [1, 2, 4]  # Multiple scales for vessel detection
        self.vessel_atts = nn.ModuleList([
            VesselAttention(in_channels) for _ in self.scales
        ])
        
        # Fusion conv
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(self.scales), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        outputs = []
        for i, scale in enumerate(self.scales):
            if scale > 1:
                scaled = nn.functional.interpolate(x, scale_factor=1/scale, 
                                                mode='bilinear', align_corners=True)
            else:
                scaled = x
                
            att_out = self.vessel_atts[i](scaled)
            
            if scale > 1:
                att_out = nn.functional.interpolate(att_out, size=x.shape[2:], 
                                                  mode='bilinear', align_corners=True)
                
            outputs.append(att_out)
        
        concat_out = torch.cat(outputs, dim=1)
        fused = self.fusion(concat_out)
        out = self.gamma * fused + x
        return out

# For backward compatibility
PAM_Module = MultiScaleVesselAttention
CAM_Module = VesselAttention

