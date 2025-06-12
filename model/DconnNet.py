import math
from model.attention import MultiScaleVesselAttention
import torch
from torchvision import models
import torch.nn as nn
from model.resnet import resnet34
# from resnet import resnet34
# import resnet
from torch.nn import functional as F
import torchsummary
from torch.nn import init
import model.gap as gap
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = conv_block(in_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class DconnNet(nn.Module):
    def __init__(self, num_class=1, input_channels=3):
        super(DconnNet, self).__init__()
        
        # Encoder
        self.backbone = resnet34(pretrained=True)
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                          stride=2, padding=3, bias=False)
        
        # Multi-scale vessel attention modules
        self.attn1 = MultiScaleVesselAttention(64)    # For layer1 features
        self.attn2 = MultiScaleVesselAttention(128)   # For layer2 features
        self.attn3 = MultiScaleVesselAttention(256)   # For layer3 features
        self.attn4 = MultiScaleVesselAttention(512)   # For layer4 features
        
        # Decoder path
        self.up1 = UpBlock(512 + 256, 256)
        self.up2 = UpBlock(256 + 128, 128)
        self.up3 = UpBlock(128 + 64, 64)
        self.up4 = UpBlock(64 + 64, 32)
        
        # Deep supervision paths
        self.deep_sup1 = nn.Conv2d(256, num_class, 1)
        self.deep_sup2 = nn.Conv2d(128, num_class, 1)
        self.deep_sup3 = nn.Conv2d(64, num_class, 1)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            conv_block(32, 32),
            nn.Conv2d(32, num_class, kernel_size=1)
        )

    def forward(self, x):
        # Get input size
        input_size = x.size()[2:]
        
        # Initial convolution
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x0 = self.backbone.relu(x)
        x = self.backbone.maxpool(x0)
        
        # Encoder path with multi-scale vessel attention
        x1 = self.backbone.layer1(x)
        x1_att = self.attn1(x1)
        
        x2 = self.backbone.layer2(x1)
        x2_att = self.attn2(x2)
        
        x3 = self.backbone.layer3(x2)
        x3_att = self.attn3(x3)
        
        x4 = self.backbone.layer4(x3)
        x4_att = self.attn4(x4)
        
        # Decoder path with deep supervision
        d1 = self.up1(x4_att, x3_att)
        d1_out = self.deep_sup1(d1)
        d1_out = F.interpolate(d1_out, size=input_size, mode='bilinear', align_corners=True)
        
        d2 = self.up2(d1, x2_att)
        d2_out = self.deep_sup2(d2)
        d2_out = F.interpolate(d2_out, size=input_size, mode='bilinear', align_corners=True)
        
        d3 = self.up3(d2, x1_att)
        d3_out = self.deep_sup3(d3)
        d3_out = F.interpolate(d3_out, size=input_size, mode='bilinear', align_corners=True)
        
        d4 = self.up4(d3, x0)
        out = self.final_conv(d4)
        
        # Ensure output matches input size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training:
            return out, d1_out + d2_out + d3_out
        return out, out

if __name__ == "__main__":
    # Test the model
    model = DconnNet(num_class=1, input_channels=3)
    x = torch.randn(1, 3, 960, 960)  # CHASE_DB1 image size
    out_main, out_aux = model(x)
    print("Main output shape:", out_main.shape)
    print("Auxiliary output shape:", out_aux.shape)
