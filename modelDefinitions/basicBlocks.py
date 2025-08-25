import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# ===================================================================== #
# Section 1: U-Net 아키텍처를 구성하는 기본 블록들
# ===================================================================== #

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """MaxPool을 이용한 다운샘플링 후 DoubleConv 적용"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """업샘플링 후 DoubleConv 적용. 스킵 커넥션(Skip Connection)을 위한 concat 포함"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """최종 출력 채널 수를 맞추기 위한 1x1 Convolution 레이어"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# ===================================================================== #
# Section 2: attentionNet 아키텍처를 구성하는 기본 블록들
# ===================================================================== #

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def swish(x):
    return x * torch.sigmoid(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class attentionGuidedResBlock(nn.Module):
    def __init__(self, squeezeFilters=32, expandFilters=64, dilationRate=1, bn=False, act=False, bias=True):
        super(attentionGuidedResBlock, self).__init__()
        self.bn = bn
        self.act = act
        self.depthAtten = SELayer(squeezeFilters)
        self.exConv = nn.Conv2d(squeezeFilters, expandFilters, 1, dilation=dilationRate, padding=0, bias=bias)
        self.exConvBn = nn.BatchNorm2d(expandFilters)
        self.sepConv = SeparableConv2d(expandFilters, expandFilters, kernel_size=3, stride=1, dilation=dilationRate, padding=dilationRate, bias=bias)
        self.sepConvBn = nn.BatchNorm2d(expandFilters)
        self.sqConv = nn.Conv2d(expandFilters, squeezeFilters, 1, dilation=dilationRate, padding=0, bias=bias)
        self.sqConvBn = nn.BatchNorm2d(squeezeFilters)
    def forward(self, inputTensor):
        xDA = self.depthAtten(inputTensor)
        if self.bn:
            xEx = F.leaky_relu(self.exConvBn(self.exConv(inputTensor)))
            xSp = F.leaky_relu(self.sepConvBn(self.sepConv(xEx)))
            xSq = self.sqConvBn(self.sqConv(xSp))
        else:
            xEx = F.leaky_relu(self.exConv(inputTensor))
            xSp = F.leaky_relu(self.sepConv(xEx))
            xSq = self.sqConv(xSp)
        return inputTensor + xSq + xDA

class pixelShuffleUpsampling(nn.Module):
    def __init__(self, inputFilters, scailingFactor=2):
        super(pixelShuffleUpsampling, self).__init__()
        self.upSample = nn.Sequential(
            nn.Conv2d(inputFilters, inputFilters * (scailingFactor**2), 3, 1, 1),
            nn.BatchNorm2d(inputFilters * (scailingFactor**2)),
            nn.PixelShuffle(upscale_factor=scailingFactor),
            nn.PReLU()
        )
    def forward(self, tensor):
        return self.upSample(tensor)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialAttentionBlock(nn.Module):
    def __init__(self, spatial_filter=32):
        super(SpatialAttentionBlock, self).__init__()
        self.spatialAttenton = SpatialAttention()
        self.conv = nn.Conv2d(spatial_filter, spatial_filter, 3, padding=1)
    def forward(self, x):
        x1 = self.spatialAttenton(x)
        xC = self.conv(x)
        y = x1 * xC
        return y
