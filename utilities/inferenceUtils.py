import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
from PIL import ImageFile
# ⚠️ 1. 학습 때 사용했던 불량 화소 생성기를 가져옵니다.
from dataTools.badPixelGenerator import generate_bad_pixels 

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AddGaussianNoise(object):
    # 이 클래스는 더 이상 사용되지 않지만, 다른 곳에서 호출할 수 있으므로 그대로 둡니다.
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel

    def __call__(self, tensor):
        if self.noiseLevel == 0:
            return tensor
        sigma = self.noiseLevel/100.
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, resize = None, validation = None ):
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()

    def inputForInference(self, imagePath, noiseLevel):
        # ========================================================== #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # ---------------------------------------------------------- #
        # 기존: 깨끗한 이미지 로드 -> 약한 노이즈 추가
        # 변경: 깨끗한 이미지 로드 -> '불량 화소' 알고리즘 적용
        # ========================================================== #

        # 1. 테스트할 깨끗한 원본 이미지를 불러옵니다.
        clean_image_pil = Image.open(imagePath).convert("RGB")
        
        # 2. Numpy 배열로 변환합니다.
        clean_image_np = np.array(clean_image_pil)
        
        # 3. 학습 때와 동일한 'generate_bad_pixels' 함수를 호출하여
        #    실시간으로 불량 화소를 적용합니다.
        corrupted_image_np = generate_bad_pixels(clean_image_np)
        
        # 4. 처리된 이미지를 다시 PIL 이미지 형식으로 변환합니다.
        #    이 손상된 이미지가 이제 모델의 실제 입력이 됩니다.
        img = Image.fromarray(corrupted_image_np)

        # 5. 이미지를 텐서로 변환하고 정규화합니다.
        #    (AddGaussianNoise는 이제 사실상 의미가 없지만, 호환성을 위해 남겨둡니다.)
        transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd),
                                        AddGaussianNoise(noiseLevel=noiseLevel)])

        testImg = transform(img).unsqueeze(0)

        return testImg


    def saveModelOutput(self, modelOutput, inputImagePath, noiseLevel, step = None, ext = ".png"):
        datasetName = os.path.basename(os.path.dirname(inputImagePath))
        if step:
            imageSavingPath = os.path.join(self.outputRootDir, self.modelName, datasetName, f"{extractFileName(inputImagePath, True)}_sigma_{noiseLevel}_{self.modelName}_{step}{ext}")
        else:
            imageSavingPath = os.path.join(self.outputRootDir, self.modelName, datasetName, f"{extractFileName(inputImagePath, True)}_sigma_{noiseLevel}_{self.modelName}{ext}")
        
        os.makedirs(os.path.dirname(imageSavingPath), exist_ok=True)
        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)

    def testingSetProcessor(self):
        testSets = glob.glob(os.path.join(self.inputRootDir, '*/'))
        if not testSets: 
             testSets = [self.inputRootDir]
             
        if self.validation:
            testSets = testSets[:1]

        testImageList = []
        for t in testSets:
            testSetName = os.path.basename(os.path.normpath(t))
            createDir(os.path.join(self.outputRootDir, self.modelName, testSetName))
            imgInTargetDir = imageList(t, False)
            testImageList.extend(imgInTargetDir)
            
        return testImageList