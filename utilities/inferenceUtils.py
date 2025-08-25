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
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, validation=None, inferenceMode=2):
        self.gridSize = gridSize # Demosaic 모드 구분을 위해 사용
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.modelName = modelName
        self.validation = validation
        self.unNormalize = UnNormalize()
        self.inferenceMode = inferenceMode

    def inputForInference(self, imagePath, noiseLevel):
        source_image_pil = Image.open(imagePath).convert("RGB")

        # --- Demosaic 모드 (gridSize == -1) 처리 ---
        if self.gridSize == -1:
            print("Mode 3 (Demosaicing): Treating input as clean Quad Bayer.")
            img = source_image_pil
        # --- BP Correction 모드 처리 ---
        else:
            if self.inferenceMode == 1:
                print("Mode 1: Treating input as already defective.")
                img = source_image_pil
            elif self.inferenceMode == 2:
                print("Mode 2: Applying bad pixels to clean input.")
                source_image_np = np.array(source_image_pil)
                corrupted_image_np = generate_bad_pixels(source_image_np)
                img = Image.fromarray(corrupted_image_np)
            else:
                raise ValueError(f"Invalid inferenceMode: {self.inferenceMode}")

        # 텐서 변환 및 정규화
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normMean, normStd)
        ])
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
