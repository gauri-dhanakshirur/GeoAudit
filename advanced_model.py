import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    """
    Attention Gate: Filters features from the skip connection (x) using the 
    gating signal (g) from the coarser scale.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class StripPooling(nn.Module):
    """
    Strip Pooling: Captures long-range dependencies using 1xN and Nx1 pooling.
    Useful for linear structures like roads.
    """
    def __init__(self, in_channels, pool_size, norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = self.conv2_1(x2)
        x2_3 = self.conv2_2(x1 + x2)
        x2_4 = self.conv2_3(x1)
        x2_5 = self.conv2_4(x2)
        x1 = self.conv2_5(F.interpolate(x2_1 + x2_3 + x2_4, (h, w), mode='bilinear', align_corners=True))
        x2 = self.conv2_6(F.interpolate(x2_2 + x2_3 + x2_5, (h, w), mode='bilinear', align_corners=True))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class Dblock(nn.Module):
    """
    D-Block: Center block from D-LinkNet using dilated convolutions 
    to expand receptive field without resolution loss.
    """
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d3 = self.relu(self.dilate3(d2))
        d4 = self.relu(self.dilate4(d3))
        out = x + d1 + d2 + d3 + d4
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle padding issues if dimensions don't match exactly
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class AdvancedRoadModel(nn.Module):
    def __init__(self, num_classes=1):
        super(AdvancedRoadModel, self).__init__()
        
        # 1. Encoder: ResNet34
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Extract layers
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1 # 64
        self.layer2 = self.backbone.layer2 # 128
        self.layer3 = self.backbone.layer3 # 256
        self.layer4 = self.backbone.layer4 # 512
        
        # 2. D-Block (Center) - D-LinkNet style
        self.dblock = Dblock(512)
        
        # 3. Strip Pooling (Optional integration - implemented here at center for robust features)
        self.strip_pool = StripPooling(512, (20, 12)) 

        # 4. Filter counts for decoder
        filters = [64, 128, 256, 512]
        
        # 5. Attention Gates
        # Gating signal 'g' comes from the upsampled feature map. 
        # Skip connection 'x' comes from the encoder.
        
        # for att4: 
        # center(512) -> up4 -> d4(256). So g has 256 channels.
        # x3 has 256 channels.
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        
        # for att3:
        # d4(256) -> up3 -> d3(128). So g has 128 channels.
        # x2 has 128 channels.
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        
        # for att2:
        # d3(128) -> up2 -> d2(64). So g has 64 channels.
        # x1 has 64 channels.
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Decoder 4: Up from 512, skip 256
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        # Decoder 3: Up from 256, skip 128
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128+128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

        # Decoder 2: Up from 128, skip 64
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        
        # Decoder 1: Up from 64, skip ? 
        # ResNet layer1 is same res as conv1 output (1/4 or 1/2 depending on implementation).
        # We need to get back to full size.
        # layer1 is 64 channels, H/4, W/4.
        
        # Let's refine the last blocks to reach original size.
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # H/2
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # H
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        # Encoder
        x = self.layer0[0](x) # conv1
        x = self.layer0[1](x) # bn1
        x0 = self.layer0[2](x) # relu (64, H/2, W/2)
        x_pool = self.layer0[3](x0) # maxpool (64, H/4, W/4)
        
        x1 = self.layer1(x_pool) # (64, H/4, W/4)
        x2 = self.layer2(x1) # (128, H/8, W/8)
        x3 = self.layer3(x2) # (256, H/16, W/16)
        x4 = self.layer4(x3) # (512, H/32, W/32)
        
        # Center: D-Block + Strip Pooling
        center = self.dblock(x4)
        center = self.strip_pool(center)
        
        # Decoder with Attention
        
        # Dec 4 (x4 -> x3)
        d4 = self.up4(center) # 256
        x3 = self.att4(g=d4, x=x3)
        if d4.size() != x3.size():
             d4 = F.interpolate(d4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.dec4(d4)
        
        # Dec 3 (d4 -> x2)
        d3 = self.up3(d4) # 128
        x2 = self.att3(g=d3, x=x2)
        if d3.size() != x2.size():
             d3 = F.interpolate(d3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.dec3(d3)
        
        # Dec 2 (d3 -> x1)
        d2 = self.up2(d3) # 64
        x1 = self.att2(g=d2, x=x1)
        if d2.size() != x1.size():
             d2 = F.interpolate(d2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.dec2(d2)
        
        # Final Upsampling
        d1 = self.up1(d2) # 32
        out = self.final(d1)
        
        return out
