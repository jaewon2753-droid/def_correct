import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.badPixelGenerator import generate_bad_pixels
import os

class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)

        # 사용자 요청: 이미지 리사이즈는 하지 않음 (원본 크기 유지)
        self.transformHRGT = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.transformRI = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # 1) GT 로드
        try:
            gt_image_pil = Image.open(self.image_list[i]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {self.image_list[i]}: {e}")
            return self.__getitem__((i + 1) % len(self.image_list))

        # 2) 불량 화소 생성 (입력 이미지 제작)
        gt_image_np = np.array(gt_image_pil)
        input_image_np = generate_bad_pixels(gt_image_np)
        input_image_pil = Image.fromarray(input_image_np)

        # 3) 변환 적용 (리사이즈 없음)
        input_tensor = self.transformRI(input_image_pil)
        gt_tensor = self.transformHRGT(gt_image_pil)

        return input_tensor, gt_tensor
