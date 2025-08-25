import torch
import os
from utilities.customUtils import *
from utilities.inferenceUtils import inference  # 재사용
from modelDefinitions.unet_transformer_gen import UNetTransformer  # UNet 기반으로 통일

class Demosaicer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # attentionNet 대신 UNetTransformer 사용
        self.model = UNetTransformer(n_channels=3, n_classes=3).to(self.device)

    def load_demosaic_weights(self, weight_type):
        # 디모자이킹 가중치 경로를 설정합니다.
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

        # BJDD에서 저장된 다양한 키를 모두 수용 (stateDictEG 또는 stateDictG)
        checkpoint = torch.load(weight_path, map_location=self.device)
        sd = checkpoint.get('stateDictEG') or checkpoint.get('stateDictG') or checkpoint
        if isinstance(sd, dict):
            self.model.load_state_dict(sd, strict=False)
        else:
            raise KeyError("No compatible state dict found in checkpoint.")
        print("Demosaicing weight loaded successfully.")

    def run_demosaic(self, input_dir, output_dir, weight_type):
        self.load_demosaic_weights(weight_type)
        self.model.eval()

        # inference 유틸을 재사용하되 gridSize=-1 로 디모자이크 모드 표시
        modelInference = inference(
            gridSize=-1,               # Demosaicing 모드 플래그
            inputRootDir=input_dir,
            outputRootDir=output_dir,
            modelName=f"Demosaic_{weight_type}",
            inferenceMode=3,           # 모드 표시(로깅 목적)
        )

        testImageList = modelInference.testingSetProcessor()
        with torch.no_grad():
            for imgPath in testImageList:
                img = modelInference.inputForInference(imgPath, noiseLevel=0).to(self.device)
                output = self.model(img)
                modelInference.saveModelOutput(output, imgPath, noiseLevel=0)
        print("\nDemosaicing completed!")
