# mainModule/Demosaicer.py
import torch
import os
import glob
from PIL import Image
from utilities.customUtils import *
from utilities.inferenceUtils import inference # 일부 기능 재사용
from modelDefinitions.attentionGen_bjdd import attentionNet # BJDD의 원본 모델

class Demosaicer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = attentionNet().to(self.device)

    def load_demosaic_weights(self, weight_type):
        # Demosaicing을 위한 가중치 경로를 설정합니다.
        # 이 폴더에 'original_bjdd.pth', 'custom_demosaic.pth' 파일을 미리 넣어두어야 합니다.
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

        # BJDD 모델은 stateDictEG 키를 사용했으므로, 해당 키로 가중치를 불러옵니다.
        checkpoint = torch.load(weight_path)
        self.model.load_state_dict(checkpoint['stateDictEG'])
        print("Demosaicing weight loaded successfully.")

    def run_demosaic(self, input_dir, output_dir, weight_type):
        self.load_demosaic_weights(weight_type)

        # inference 클래스를 재사용하여 이미지 목록을 가져오고 결과를 저장합니다.
        # gridSize=-1 과 같은 특별한 값을 주어 Demosaicing 모드임을 알립니다.
        modelInference = inference(
            gridSize=-1, # Demosaicing 모드 플래그
            inputRootDir=input_dir,
            outputRootDir=output_dir,
            modelName=f"Demosaic_{weight_type}"
        )

        testImageList = modelInference.testingSetProcessor()
        with torch.no_grad():
            for imgPath in testImageList:
                # Demosaicing은 추가 노이즈나 BP 생성이 필요 없습니다.
                img = modelInference.inputForInference(imgPath, noiseLevel=0).to(self.device)
                output = self.model(img)
                modelInference.saveModelOutput(output, imgPath, noiseLevel=0)
        print("\nDemosaicing completed!")
