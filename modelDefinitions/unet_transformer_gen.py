import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicBlocks import DoubleConv, Down, Up, OutConv

# (TransformerBlock 클래스는 변경 없이 그대로 사용합니다)
class TransformerBlock(nn.Module):
    def __init__(self, channels, n_heads, n_layers, patch_size=2, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.reshape_conv = nn.Conv2d(embed_dim, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_res = x
        
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2).view(b, -1, h // self.patch_size, w // self.patch_size)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.reshape_conv(x)
        
        return x + x_res

# ========================================================== #
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
# ---------------------------------------------------------- #
# U-Net의 깊이를 한 단계 더 깊게 만들어 128x128 이미지에 최적화합니다.
# 128 -> 64 -> 32 -> 16 (병목) -> 32 -> 64 -> 128
# ========================================================== #
class UNetTransformer(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetTransformer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- 인코더 (한 층 추가) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512) # <-- 새로 추가된 다운샘플링 레이어
        
        # --- 트랜스포머 병목 ---
        # 128x128 이미지가 3번 다운샘플링되면 16x16이 됨
        # 채널은 512
        self.transformer = TransformerBlock(channels=512, n_heads=8, n_layers=4, patch_size=2)
                                             
        # --- 디코더 (한 층 추가) ---
        self.up1 = Up(512, 256) # <-- 새로 추가된 업샘플링 레이어
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        input_image = x
        
        # 인코더 경로 (한 층 깊어짐)
        x1 = self.inc(input_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # <-- 추가
        
        # 트랜스포머 병목 구간
        x_transformer = self.transformer(x4)
        
        # 디코더 경로 (스킵 커넥션 연결 수정)
        x = self.up1(x_transformer, x3) # <-- x3와 연결
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        return torch.tanh(logits + input_image)