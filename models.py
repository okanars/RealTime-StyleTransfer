import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple

class ConvBlock(nn.Module):
    """Helper class for Conv Layer + Instance Norm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm="instance"):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm == "instance" else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        return self.relu(x)

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        return out + residual

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvBlock(3, 32, kernel_size=9, stride=1, padding=4)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)

        # Residual blocks (Bottleneck)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Upsampling Layers
        # We use Upsample + Conv instead of ConvTranspose to avoid checkerboard artifacts
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = ConvBlock(64, 32, kernel_size=3, stride=1, padding=1)

        # Output layer
        self.deconv3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.deconv1(x)
        x = self.up2(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        # Tanh is generally used in style transfer outputs (scaled later)
        # But we will rely on raw output and normalize in utils usually, 
        # let's stick to standard practice: no activation at very end or Sigmoid/Tanh depending on scaling.
        # Here we return raw logits, training loop handles limits.
        return x

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Slice VGG to get intermediate layers for style/content loss
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out