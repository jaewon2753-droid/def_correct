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

un = UnNormalize()

def findLastWeights(path, modelName=None):
    """
    지정된 경로에서 가장 최신 체크포인트 파일을 찾습니다.
    """
    # 검색할 파일 패턴을 생성합니다. 예: /path/to/weights/MyModel_checkpoint*.pth
    search_pattern = os.path.join(path, f"{modelName}_checkpoint*.pth")
    previousWeights = glob.glob(search_pattern)

    if not previousWeights:
        print("Checkpoint directory is empty")
        return None

    # 파일 이름에서 스텝(step) 번호를 추출하여 최신 파일을 찾습니다.
    latest_file = max(previousWeights, key=lambda p: int(p.split('_')[-1].split('.')[0]) if p.split('_')[-1].split('.')[0].isdigit() else -1)
    
    print(f"Found latest checkpoint: {latest_file}")
    return latest_file


def saveCheckpoint(modelStates, path, modelName=None, currentStep=None, backup=True):
    if not modelName:
        modelName = "model"
    
    createDir(path)

    # 저장할 파일 이름을 구성합니다. 예: /path/to/weights/MyModel_checkpoint_100.pth
    if currentStep:
        cpName = os.path.join(path, f"{modelName}_checkpoint_{str(currentStep)}.pth")
    else:
        cpName = os.path.join(path, f"{modelName}_checkpoint.pth")

    torch.save(modelStates, cpName)

def loadCheckpoints(path, modelName, epoch=False, lastWeights=True):
    if lastWeights:
        cpPath = findLastWeights(path, modelName)
    else:
        # 특정 에포크를 로드하는 로직 (필요 시 구현)
        # 현재는 lastWeights=True만 사용되므로 이 부분은 덜 중요합니다.
        cpPath = os.path.join(path, f"{modelName}_checkpoint.pth")

    if cpPath is None or not os.path.exists(cpPath):
        raise FileNotFoundError(f"Checkpoint file not found. Searched in path: {path} with model name: {modelName}")

    checkpoint = torch.load(cpPath)
    return checkpoint

def tbLogWritter(summaryInfo):
    log_path = summaryInfo.get('Path', './logDir/')
    epoch = summaryInfo.get('Epoch', 0)
    step = summaryInfo.get('Step', 0)

    writer_path = os.path.join(log_path, f"epoch_{epoch}")
    createDir(log_path)
    writer = SummaryWriter(writer_path)

    for k, v in summaryInfo.items():
        if 'Image' in k:
            writer.add_image(k, torchvision.utils.make_grid(v), step)
        elif 'Loss' in k:
            writer.add_scalar(k, v, step)

    writer.close()