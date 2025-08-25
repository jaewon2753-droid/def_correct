# mainModule/Demosaicer.py

import torch
import os
from utilities.customUtils import *
from utilities.inferenceUtils import inference
# ========================================================== #
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
# ---------------------------------------------------------- #
# UNetTransformer 대신, Demosaicing을 위한 attentionNet 모델을
# attentionGen_bjdd.py 파일에서 정확하게 불러옵니다.
# ========================================================== #
from modelDefinitions.attentionGen_bjdd import attentionNet

class Demosaicer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 모델을 attentionNet으로 올바르게 수정합니다.
        self.model = attentionNet().to(self.device)

    def load_demosaic_weights(self, weight_type):
        weight_dir = "./demosaic_weights/"
        if weight_type == "original":
            weight_path = os.path.join(weight_dir, "original_bjdd.pth")
        elif weight_type == "custom":
            weight_path = os.path.join(weight_dir, "custom_demosaic.pth")
        else:
            raise ValueError("Invalid demosaic weight type")

        print(f"Loading Demosaicing weight from: {weight_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        # BJDD 모델은 stateDictEG 또는 stateDictG 키를 사용하므로, 둘 다 확인하여 불러옵니다.
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = checkpoint.get('stateDictEG') or checkpoint.get('stateDictG')

        if state_dict:
            # strict=False 옵션은 일부 키가 맞지 않더라도 최대한 불러오도록 허용합니다.
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # .pth 파일이 가중치만 바로 저장된 경우를 대비
            self.model.load_state_dict(checkpoint, strict=False)

        print("Demosaicing weight loaded successfully.")

    def run_demosaic(self, input_dir, output_dir, weight_type):
        self.load_demosaic_weights(weight_type)
        self.model.eval() # 추론 모드로 설정

        modelInference = inference(
            gridSize=-1,
            inputRootDir=input_dir,
            outputRootDir=output_dir,
            modelName=f"Demosaic_{weight_type}",
            inferenceMode=3
        )

        testImageList = modelInference.testingSetProcessor()
        with torch.no_grad():
            for imgPath in testImageList:
                img = modelInference.inputForInference(imgPath, noiseLevel=0).to(self.device)
                output = self.model(img)
                modelInference.saveModelOutput(output, imgPath, noiseLevel=0)
        print("\nDemosaicing completed!")
