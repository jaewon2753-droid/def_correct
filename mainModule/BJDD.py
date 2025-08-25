import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import sys
import glob
import time
import colorama
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from torchsummary import summary
from ptflops import get_model_complexity_info
from utilities.torchUtils import *
from dataTools.customDataloader import *
from utilities.inferenceUtils import *
from utilities.aestheticUtils import *
from loss.pytorch_msssim import *
from loss.colorLoss import *
from loss.percetualLoss import *

# ========================================================== #
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 모델 임포트 변경 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
# ========================================================== #
# 기존 attentionGen 대신 새로 만든 unet_transformer_gen을 가져옵니다.
from modelDefinitions.unet_transformer_gen import UNetTransformer
from modelDefinitions.attentionDis import *
# ========================================================== #
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
# ========================================================== #

from torchvision.utils import save_image


class BJDD:
    def __init__(self, config):

        # --- config.json 파일로부터 모델 설정을 불러옵니다 ---
        self.gtPath = config['gtPath']
        self.targetPath = config['targetPath']
        self.checkpointPath = config['checkpointPath']
        self.logPath = config['logPath']
        self.testImagesPath = config['testImagePath']
        self.resultDir = config['resultDir']
        self.modelName = config['modelName']
        self.dataSamples = config['dataSamples']
        self.batchSize = int(config['batchSize'])
        self.imageH = int(config['imageH'])
        self.imageW = int(config['imageW'])
        self.inputC = int(config['inputC'])
        self.outputC = int(config['outputC'])
        self.totalEpoch = int(config['epoch'])
        self.interval = int(config['interval'])
        self.learningRate = float(config['learningRate'])
        self.adamBeta1 = float(config['adamBeta1'])
        self.adamBeta2 = float(config['adamBeta2'])
        self.barLen = int(config['barLen'])

        # 학습 진행 상태를 위한 변수 초기화
        self.currentEpoch = 0
        self.startSteps = 0
        self.totalSteps = 0

        # 이미지 정규화 해제를 위한 클래스 인스턴스
        self.unNorm = UnNormalize()

        # 추론 시 사용할 노이즈 레벨 (불량 화소 복원 프로젝트에서는 큰 의미 없음)
        self.noiseSet = [0, 5, 10]

        # GPU 사용 설정
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ========================================================== #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 생성자 모델 변경 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # ========================================================== #
        # self.attentionNet = attentionNet().to(self.device) # 기존 모델
        self.generator = UNetTransformer(n_channels=self.inputC, n_classes=self.outputC).to(self.device)
        # ========================================================== #
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
        # ========================================================== #

        self.discriminator = attentiomDiscriminator().to(self.device)

        # 옵티마이저(Optimizer) 정의
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.learningRate,
                                           betas=(self.adamBeta1, self.adamBeta2))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate,
                                           betas=(self.adamBeta1, self.adamBeta2))

        # 스케줄러 (필요 시 사용)
        self.scheduleLR = None

    def customTrainLoader(self, overFitTest=False):

        # targetPath (GT 이미지가 있는 곳)에서 이미지 목록을 불러옴
        targetImageList = imageList(self.targetPath)
        print("Trining Samples (GT):", self.targetPath, len(targetImageList))

        if overFitTest == True:
            targetImageList = targetImageList[-1:]
        if self.dataSamples:
            targetImageList = targetImageList[:int(self.dataSamples)]

        datasetReadder = customDatasetReader(
            image_list=targetImageList,
            imagePathGT=self.gtPath,
            height=self.imageH,
            width=self.imageW,
        )

        self.trainLoader = torch.utils.data.DataLoader(dataset=datasetReadder,
                                                       batch_size=self.batchSize,
                                                       shuffle=True,
                                                       num_workers=2  # 데이터 로딩 속도 향상
                                                       )

        return self.trainLoader

    def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples=None):

        if dataSamples:
            self.dataSamples = dataSamples

            # --- 손실 함수(Loss Function) 정의 ---
        # L1 Loss: 픽셀 단위의 정확한 복원을 위함
        reconstructionLoss = torch.nn.L1Loss().to(self.device)
        # Perceptual Loss: 이미지의 전반적인 질감과 특징을 비슷하게 만듦
        featureLoss = regularizedFeatureLoss().to(self.device)
        # Color Loss: 색감을 비슷하게 만듦
        colorLoss = deltaEColorLoss(normalize=True).to(self.device)
        # Adversarial Loss: GAN 학습을 위함
        adversarialLoss = nn.BCELoss().to(self.device)

        # 데이터 로더 준비
        trainingImageLoader = self.customTrainLoader(overFitTest=overFitTest)

        # 이어서 학습하기 (Resume Training)
        if resumeTraning == True:
            try:
                self.modelLoad()
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                customPrint(Fore.RED + "Starting training from scratch.", textWidth=self.barLen)

        # --- 본격적인 학습 루프 시작 ---
        customPrint('Training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(),
                    textWidth=self.barLen)

        self.totalSteps = int(len(trainingImageLoader) * self.totalEpoch)
        startTime = time.time()

        bar = ProgressBar(self.totalSteps, max_width=int(self.barLen / 2))
        currentStep = self.startSteps
        while currentStep < self.totalSteps:
            for i, (inputImages, gtImages) in enumerate(trainingImageLoader):

                # --- 변수 준비 ---
                currentStep += 1
                if currentStep > self.totalSteps:
                    break

                # 이미지를 GPU로 이동
                input_real = inputImages.to(self.device)
                gt_real = gtImages.to(self.device)

                # GAN 학습을 위한 정답 라벨 (Label Smoothing 적용)
                target_real_label = (torch.rand(input_real.shape[0], 1) * 0.3 + 0.7).to(self.device)
                target_fake_label = (torch.rand(input_real.shape[0], 1) * 0.3).to(self.device)
                target_ones_label = torch.ones(input_real.shape[0], 1).to(self.device)

                # --- 1. 판별자(Discriminator) 훈련 ---
                self.optimizerD.zero_grad()

                # 생성자가 만든 가짜 이미지
                generated_fake = self.generator(input_real)

                # 진짜 이미지는 '진짜'로, 가짜 이미지는 '가짜'로 판별하도록 학습
                lossD = adversarialLoss(self.discriminator(gt_real), target_real_label) + \
                        adversarialLoss(self.discriminator(generated_fake.detach()), target_fake_label)
                lossD.backward()
                self.optimizerD.step()

                # --- 2. 생성자(Generator) 훈련 ---
                self.optimizerG.zero_grad()

                # 목표 1: 정답(GT) 이미지와 최대한 비슷해지기
                lossG_content = reconstructionLoss(generated_fake, gt_real) + \
                                featureLoss(generated_fake, gt_real) + \
                                colorLoss(generated_fake, gt_real)

                # 목표 2: 판별자를 속이기 (자신이 만든 가짜를 진짜라고 믿게 만들기)
                lossG_adversarial = adversarialLoss(self.discriminator(generated_fake), target_ones_label)

                # 두 목표를 합쳐 최종 생성자 손실 계산
                lossG = lossG_content + 1e-3 * lossG_adversarial
                lossG.backward()
                self.optimizerG.step()

                # --- 로그 기록 및 가중치 저장 ---
                if (currentStep + 1) % self.interval == 0:
                    # Tensorboard 로그 기록
                    summaryInfo = {
                        'Input Images': self.unNorm(input_real),
                        'Generated Images': self.unNorm(generated_fake),
                        'GT Images': self.unNorm(gt_real),
                        'Step': currentStep + 1,
                        'Epoch': self.currentEpoch,
                        'LossG': lossG.item(),
                        'LossD': lossD.item(),
                        'Path': self.logPath,
                    }
                    tbLogWritter(summaryInfo)

                    # 현재 가중치 저장
                    self.savingWeights(currentStep)

            self.currentEpoch += 1

        # 최종 가중치 저장
        self.savingWeights(currentStep, duplicate=True)
        customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)

    def modelInference(self, testImagesPath=None, outputDir=None, resize=None, validation=None, noiseSet=None,
                       steps=None):
        if not validation:
            self.modelLoad()
            print("\nInferencing on pretrained weights.")

        if not noiseSet:
            noiseSet = self.noiseSet
        if testImagesPath:
            self.testImagesPath = testImagesPath
        if outputDir:
            self.resultDir = outputDir

        modelInference = inference(gridSize=0, inputRootDir=self.testImagesPath, outputRootDir=self.resultDir,
                                   modelName=self.modelName, validation=validation)

        testImageList = modelInference.testingSetProcessor()
        with torch.no_grad():
            for noise in noiseSet:
                for imgPath in testImageList:
                    img = modelInference.inputForInference(imgPath, noiseLevel=noise).to(self.device)
                    # 생성자 모델로 복원 실행
                    output = self.generator(img)
                    modelInference.saveModelOutput(output, imgPath, noise, steps)
        print("\nInference completed!")

    def modelSummary(self, input_size=None):
        if not input_size:
            input_size = (self.inputC, self.imageH, self.imageW)

        customPrint(Fore.YELLOW + "Generator (U-Net Transformer)", textWidth=self.barLen)
        summary(self.generator, input_size=input_size)
        print("*" * self.barLen)
        print()

        customPrint(Fore.YELLOW + "Discriminator", textWidth=self.barLen)
        summary(self.discriminator, input_size=input_size)
        print("*" * self.barLen)
        print()
        configShower()

    def savingWeights(self, currentStep, duplicate=None):
        checkpoint = {
            'step': currentStep + 1,
            'stateDictG': self.generator.state_dict(),
            'stateDictD': self.discriminator.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
        }
        saveCheckpoint(modelStates=checkpoint, path=self.checkpointPath, modelName=self.modelName)
        if duplicate:
            saveCheckpoint(modelStates=checkpoint, path=self.checkpointPath + "backup_" + str(currentStep) + "/",
                           modelName=self.modelName, backup=None)

    def modelLoad(self):
        customPrint(Fore.RED + "Loading pretrained weight", textWidth=self.barLen)

        previousWeight = loadCheckpoints(self.checkpointPath, self.modelName)
        self.generator.load_state_dict(previousWeight['stateDictG'])
        self.discriminator.load_state_dict(previousWeight['stateDictD'])
        self.optimizerG.load_state_dict(previousWeight['optimizerG'])
        self.optimizerD.load_state_dict(previousWeight['optimizerD'])
        self.startSteps = int(previousWeight['step'])

        customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=self.barLen)