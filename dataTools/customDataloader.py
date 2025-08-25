import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
# 새로 만든 불량 화소 생성기를 가져옵니다.
from dataTools.badPixelGenerator import generate_bad_pixels
import os


class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        # gtPath와 targetPath가 동일하므로 imagePathGT는 사실상 사용되지 않습니다.
        # image_list 자체가 GT 이미지 목록입니다.
        self.imagePathGT = imagePathGT
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)

        # 정답(GT) 이미지용 변환: 텐서(Tensor)로 변환하고 정규화합니다.
        self.transformHRGT = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # 모델 입력(Input) 이미지용 변환: 텐서로 변환하고 정규화합니다.
        self.transformRI = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        # 전체 데이터셋의 길이를 반환합니다.
        return (len(self.image_list))

    def __getitem__(self, i):
        """
        데이터셋에서 하나의 샘플(입력 이미지, 정답 이미지 쌍)을 가져옵니다.
        """

        # 1. 원본(GT) 이미지 불러오기
        # targetPath와 gtPath가 같으므로, image_list에서 바로 GT 이미지를 불러옵니다.
        try:
            # 이미지를 열고 RGB 형식으로 변환하여 채널 수를 통일합니다.
            gt_image_pil = Image.open(self.image_list[i]).convert("RGB")
        except Exception as e:
            # 파일 읽기 오류 발생 시, 오류를 출력하고 다음 이미지를 가져옵니다.
            print(f"Error loading image {self.image_list[i]}: {e}")
            # 재귀 호출을 통해 다음 인덱스의 아이템을 반환합니다.
            return self.__getitem__((i + 1) % len(self.image_list))

        # 2. GT 이미지를 Numpy 배열로 변환
        # 불량 화소 생성을 위해 이미지 데이터를 배열 형태로 바꿉니다.
        gt_image_np = np.array(gt_image_pil)

        # 3. 불량 화소 생성기를 사용하여 모델에 입력할 이미지 생성
        # 원본 GT 이미지에 실시간으로 다양한 불량 화소를 적용합니다.
        input_image_np = generate_bad_pixels(gt_image_np)

        # 4. Numpy 배열을 다시 PIL 이미지 형식으로 변환
        # Pytorch의 transforms를 적용하기 위해 다시 이미지 객체로 만듭니다.
        input_image_pil = Image.fromarray(input_image_np)

        # 5. 각각의 이미지에 정의된 변환(transform)을 적용
        # 모델이 처리할 수 있는 텐서 형태로 최종 변환합니다.
        self.inputImage = self.transformRI(input_image_pil)
        self.gtImageHR = self.transformHRGT(gt_image_pil)

        # 최종적으로 (입력 이미지 텐서, 정답 이미지 텐서) 쌍을 반환합니다.
        return self.inputImage, self.gtImageHR
