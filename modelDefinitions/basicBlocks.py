import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# U-Net 아키텍처를 구성하는 기본 블록들
# --------------------------------------------------------------------------- #

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
        # ConvTranspose2d를 사용하여 업샘플링 (Upsampling)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 업샘플링할 텐서 (깊은 층에서 온 것)
        # x2: 스킵 커넥션으로 연결될 텐서 (얕은 층에서 온 것)
        x1 = self.up(x1)

        # x2와 x1의 크기가 홀/짝 문제로 1픽셀 차이날 경우를 대비해 패딩(padding)으로 크기를 맞춰줌
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 채널(channel) 차원을 기준으로 두 텐서를 합침 (concatenate)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """최종 출력 채널 수를 맞추기 위한 1x1 Convolution 레이어"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)