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

# 손실들
from loss.colorLoss import ColorLoss
from loss.percetualLoss import regularizedFeatureLoss
from loss.pytorch_msssim import MSSSIM  # ← 너가 추가한 모듈

# ========================================================== #
# ▼▼ 모델 임포트 ▼▼
#   - Generator: UNetTransformer  (Bad Pixel Correct)
#   - Discriminator: attentiomDiscriminator (로짓 반환! 시그모이드 없음)
# ========================================================== #
from modelDefinitions.unet_transformer_gen import UNetTransformer
from modelDefinitions.attentionDis import attentiomDiscriminator
# ========================================================== #

from torchvision.utils import save_image


class BJDD:
    def __init__(self, config):
        # --- config.json 로드 ---
        self.gtPath         = config['gtPath']
        self.targetPath     = config['targetPath']
        self.checkpointPath = config['checkpointPath']
        self.logPath        = config['logPath']
        self.testImagesPath = config['testImagePath']
        self.resultDir      = config['resultDir']
        self.modelName      = config['modelName']
        self.dataSamples    = config['dataSamples']
        self.batchSize      = int(config['batchSize'])
        self.imageH         = int(config['imageH'])
        self.imageW         = int(config['imageW'])
        self.inputC         = int(config['inputC'])
        self.outputC        = int(config['outputC'])
        self.totalEpoch     = int(config['epoch'])
        self.interval       = int(config['interval'])
        self.learningRate   = float(config['learningRate'])
        self.adamBeta1      = float(config['adamBeta1'])
        self.adamBeta2      = float(config['adamBeta2'])
        self.barLen         = int(config['barLen'])

        # 진행 상태
        self.currentEpoch = 0
        self.startSteps  = 0
        self.totalSteps  = 0

        # 정규화 해제 도우미
        self.unNorm = UnNormalize()

        # 추론용 노이즈 레벨 세트(프로젝트 성격상 의미 제한적)
        self.noiseSet = [0, 5, 10]

        # 디바이스
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ===== Generator / Discriminator =====
        self.generator     = UNetTransformer(n_channels=self.inputC, n_classes=self.outputC).to(self.device)
        self.discriminator = attentiomDiscriminator().to(self.device)  # ← forward에서 sigmoid 없음 (logits)

        # Optimizer
        self.optimizerG = torch.optim.Adam(
            self.generator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2)
        )
        self.optimizerD = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2)
        )

        self.scheduleLR = None  # (필요 시 사용)

    def customTrainLoader(self, overFitTest=False):
        # GT 목록
        targetImageList = imageList(self.targetPath)
        print("Trining Samples (GT):", self.targetPath, len(targetImageList))

        if overFitTest:
            targetImageList = targetImageList[-1:]
        if self.dataSamples:
            targetImageList = targetImageList[:int(self.dataSamples)]

        datasetReadder = customDatasetReader(
            image_list=targetImageList,
            imagePathGT=self.gtPath,
            height=self.imageH,
            width=self.imageW,
        )

        # 주의: 리사이즈 미사용이면 모든 이미지 해상도가 동일해야 batch>1 가능
        self.trainLoader = torch.utils.data.DataLoader(
            dataset=datasetReadder,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=2
        )
        return self.trainLoader

    def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples=None):
        if dataSamples:
            self.dataSamples = dataSamples

        # --- 손실 정의 ---
        reconstructionLoss = torch.nn.L1Loss().to(self.device)
        featureLoss        = regularizedFeatureLoss().to(self.device)
        colorLoss          = ColorLoss().to(self.device)
        adversarialLoss    = nn.BCEWithLogitsLoss().to(self.device)  # ← 판별기 로짓 + 이 손실 조합

        # MS-SSIM (너 구현: [-1,1] 자동 처리)
        ssimLoss    = MSSSIM(window_size=11, size_average=True, channel=self.outputC).to(self.device)
        lambda_ssim = 0.2  # 필요 시 0.1~0.3 튜닝

        # 데이터로더
        trainingImageLoader = self.customTrainLoader(overFitTest=overFitTest)

        # 이어서 학습
        if resumeTraning:
            try:
                self.modelLoad()
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                customPrint(Fore.RED + "Starting training from scratch.", textWidth=self.barLen)

        # --- 루프 시작 ---
        customPrint('Training is about to begin using:' + Fore.YELLOW + f'[{self.device}]'.upper(), textWidth=self.barLen)

        self.totalSteps = int(len(trainingImageLoader) * self.totalEpoch)
        startTime = time.time()

        bar = ProgressBar(self.totalSteps, max_width=int(self.barLen / 2))
        currentStep = self.startSteps

        while currentStep < self.totalSteps:
            for i, (inputImages, gtImages) in enumerate(trainingImageLoader):
                currentStep += 1
                if currentStep > self.totalSteps:
                    break

                input_real = inputImages.to(self.device)
                gt_real    = gtImages.to(self.device)

                # Label smoothing
                B = input_real.shape[0]
                target_real_label = (torch.rand(B, 1) * 0.3 + 0.7).to(self.device)  # 0.7 ~ 1.0
                target_fake_label = (torch.rand(B, 1) * 0.3).to(self.device)        # 0.0 ~ 0.3
                target_ones_label = torch.ones(B, 1).to(self.device)

                # ====== 1) Update Discriminator ======
                self.optimizerD.zero_grad()

                generated_fake = self.generator(input_real)

                # D는 로짓 반환, 손실은 BCEWithLogitsLoss
                lossD_real = adversarialLoss(self.discriminator(gt_real),      target_real_label)
                lossD_fake = adversarialLoss(self.discriminator(generated_fake.detach()), target_fake_label)
                lossD = lossD_real + lossD_fake
                lossD.backward()
                self.optimizerD.step()

                # ====== 2) Update Generator ======
                self.optimizerG.zero_grad()

                # 내용 손실(복원 + 지각 + 색상)
                lossG_content = reconstructionLoss(generated_fake, gt_real) \
                              + featureLoss(generated_fake, gt_real) \
                              + colorLoss(generated_fake,  gt_real)

                # MS-SSIM 추가 (유사도→손실)
                ms_ssim_val = ssimLoss(generated_fake, gt_real)  # scalar
                loss_ssim   = 1.0 - ms_ssim_val
                lossG_content = lossG_content + lambda_ssim * loss_ssim

                # 적대적 손실(판별기 속이기)
                lossG_adv = adversarialLoss(self.discriminator(generated_fake), target_ones_label)

                # 최종 G 손실
                lossG = lossG_content + 1e-3 * lossG_adv
                lossG.backward()
                self.optimizerG.step()

                # --- 로깅 & 체크포인트 ---
                if (currentStep + 1) % self.interval == 0:
                    summaryInfo = {
                        'Input Images':     self.unNorm(input_real),
                        'Generated Images': self.unNorm(generated_fake),
                        'GT Images':        self.unNorm(gt_real),
                        'Step':             currentStep + 1,
                        'Epoch':            self.currentEpoch,
                        'LossG':            float(lossG.item()),
                        'LossD':            float(lossD.item()),
                        # 추가 로그
                        'MS-SSIM':          float(ms_ssim_val.detach().mean().item()),
                        'Loss_SSIM':        float(loss_ssim.detach().mean().item()),
                        'Path':             self.logPath,
                    }
                    tbLogWritter(summaryInfo)
                    self.savingWeights(currentStep)

            self.currentEpoch += 1

        # 최종 가중치 저장
        self.savingWeights(currentStep, duplicate=True)
        customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)

    def modelInference(self, testImagesPath=None, outputDir=None, resize=None, validation=None, noiseSet=None,
                       steps=None, inferenceMode=2):
        """
        inferenceMode:
          - 1/2: Bad Pixel Correction
          - 3:   Demosaic (참고: Demosaicer 쪽에서 attentionNet/PIPNet을 사용할 계획이면 별도 경로)
        """
        if not validation:
            self.modelLoad()
            print("\nInferencing on pretrained weights.")

        if not noiseSet:
            noiseSet = self.noiseSet
        if testImagesPath:
            self.testImagesPath = testImagesPath
        if outputDir:
            self.resultDir = outputDir

        modelInference = inference(
            gridSize=0,
            inputRootDir=self.testImagesPath,
            outputRootDir=self.resultDir,
            modelName=self.modelName,
            validation=validation,
            inferenceMode=inferenceMode
        )

        testImageList = modelInference.testingSetProcessor()
        with torch.no_grad():
            for noise in noiseSet:
                for imgPath in testImageList:
                    img = modelInference.inputForInference(imgPath, noiseLevel=noise).to(self.device)
                    out = self.generator(img)
                    modelInference.saveModelOutput(out, imgPath, noise, steps)
        print("\nInference completed!")

    def modelSummary(self, input_size=None):
        if not input_size:
            input_size = (self.inputC, self.imageH, self.imageW)

        customPrint(Fore.YELLOW + "Generator (U-Net Transformer)", textWidth=self.barLen)
        summary(self.generator, input_size=input_size)
        print("*" * self.barLen); print()

        customPrint(Fore.YELLOW + "Discriminator", textWidth=self.barLen)
        summary(self.discriminator, input_size=input_size)
        print("*" * self.barLen); print()

        # 정의돼 있으면 환경/설정 출력
        try:
            configShower()
        except Exception:
            pass

    def savingWeights(self, currentStep, duplicate=None):
        checkpoint = {
            'step':       currentStep + 1,
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
