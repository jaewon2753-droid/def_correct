# loss/colorLoss.py

import torch
import torch.nn as nn

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
        # RGB to XYZ 변환을 위한 상수 행렬 (D65 조명 기준)
        # 이 값들은 GPU에 미리 올려두어 반복적인 이동을 방지합니다.
        self.register_buffer('rgb_to_xyz_matrix',
                             torch.tensor([
                                 [0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]
                             ], dtype=torch.float32))

        # D65 조명에 대한 XYZ 참조 흰색 값
        self.register_buffer('ref_xyz', 
                             torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32))

    def rgb_to_xyz(self, rgb_images):
        """ RGB 이미지를 XYZ 색 공간으로 변환합니다. (B, C, H, W) -> (B, C, H*W) """
        # 이미지의 채널, 높이, 너비를 가져옵니다.
        b, c, h, w = rgb_images.shape
        
        # (B, C, H, W) -> (B, C, H*W) 형태로 변경
        pixels = rgb_images.permute(0, 2, 3, 1).reshape(-1, 3) # (B*H*W, C)
        
        # 행렬 곱셈을 통해 XYZ로 변환
        xyz_pixels = torch.matmul(pixels, self.rgb_to_xyz_matrix.T)
        
        # 다시 원래 이미지 형태로 복원
        return xyz_pixels.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    def xyz_to_lab(self, xyz_images):
        """ XYZ 이미지를 Lab 색 공간으로 변환합니다. """
        # 참조 흰색 값으로 정규화
        normalized_xyz = xyz_images / self.ref_xyz.view(1, 3, 1, 1)

        # Lab 변환 공식 적용
        epsilon = 6/29
        f_xyz = torch.where(normalized_xyz > (epsilon**3),
                            torch.pow(normalized_xyz, 1/3),
                            (normalized_xyz / (3 * epsilon**2)) + (4/29))

        fx, fy, fz = f_xyz[:, 0, :, :], f_xyz[:, 1, :, :], f_xyz[:, 2, :, :]

        # L, a, b 채널 계산
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        # (B, C, H, W) 형태로 합치기
        return torch.stack([L, a, b], dim=1)

    def forward(self, gen_image, gt_image):
        """ 생성된 이미지와 원본 이미지의 Lab 색 공간에서의 L1 손실을 계산합니다. """
        # 입력 이미지 정규화 해제 ((0, 1) 범위로 변경)
        # 기존 코드에서는 UnNormalize를 사용했으나, 여기서는 직접 계산합니다.
        gen_image_unnorm = (gen_image * 0.5) + 0.5
        gt_image_unnorm = (gt_image * 0.5) + 0.5

        # RGB -> XYZ -> Lab 변환
        gen_lab = self.xyz_to_lab(self.rgb_to_xyz(gen_image_unnorm))
        gt_lab = self.xyz_to_lab(self.rgb_to_xyz(gt_image_unnorm))

        # Lab 색 공간에서 L1 Loss 계산 (Delta E와 유사한 효과)
        return self.l1_loss(gen_lab, gt_lab)
