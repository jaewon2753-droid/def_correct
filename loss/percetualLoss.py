import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19

class regularizedFeatureLoss(nn.Module):
    def __init__(self, percepRegulator = 1.0, tv_weight: float = 1e-4):
        super(regularizedFeatureLoss, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        self.loss = torch.nn.L1Loss()
        self.percepRegulator = percepRegulator
        self.tv_weight = tv_weight
    
    def forward(self, x, y):
        # VGG Feature Loss
        genFeature = self.feature_extractor(x)
        gtFeature = self.feature_extractor(y)
        featureLoss = self.loss(genFeature, gtFeature) * self.percepRegulator
       
        # TV loss (표준 TV: 생성이미지 x의 공간 미분)
        size = x.size()
        h_tv_diff = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_tv_diff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        tvloss = (h_tv_diff + w_tv_diff) / (size[0] * size[1] * size[2] * size[3])

        # 기존의 곱셈 결합 -> 가중 합으로 수정
        totalLoss = featureLoss + self.tv_weight * tvloss
        return totalLoss
