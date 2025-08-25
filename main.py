import sys
from utilities.parserUtils import *
from utilities.customUtils import *
from utilities.aestheticUtils import *
from mainModule.BJDD import *
import os
# 현재 파일의 위치를 기준으로 프로젝트 최상위 폴더를 파이썬 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if __name__ == "__main__":

    # 1. 사용자가 입력한 명령어 옵션을 분석합니다.
    options = mainParser(sys.argv[1:])
    if len(sys.argv) == 1:
        customPrint("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()
    
    # 2. config.json 설정 파일을 읽어옵니다.
    if options.conf:
        configCreator()
    config = configReader()
  
    # 3. 옵션에 따라 config 파일 내용을 동적으로 업데이트합니다.
    if options.epoch:
        config=updateConfig(entity='epoch', value=options.epoch)
    if options.batch:
        config=updateConfig(entity='batchSize', value=options.batch)
    if options.manualUpdate:
        config=manualUpdateEntity()

    # 4. 명령어 옵션에 따라 적절한 작업을 수행합니다.
    if options.modelSummary:
        # --- 모델 요약 정보 출력 ---
        BJDD(config).modelSummary()

    elif options.train or options.retrain:
        # --- 학습 모드 ---
        # -ts 또는 -tr 옵션이 주어지면, BJDD 클래스의 modelTraining 함수를 실행합니다.
        # 이 부분은 Bad Pixel Correction 모델을 학습시키는 데 사용됩니다.
        is_resume = options.retrain
        BJDD(config).modelTraining(resumeTraning=is_resume, dataSamples=options.dataSamples)

    elif options.inference:
        # --- 추론 모드 ---
        # -i 옵션이 주어지면, --mode 값에 따라 다른 작업을 수행합니다.
        if options.sourceDir is None:
             print("Error: Inference mode requires a source directory. Please specify with -s option.")
             exit()

        if options.mode == 1 or options.mode == 2:
            # Mode 1, 2: Bad Pixel Correction
            print(f"--- Running Bad Pixel Correction (Mode {options.mode}) ---")
            BJDD(config).modelInference(
                testImagesPath=options.sourceDir,
                outputDir=options.resultDir,
                inferenceMode=options.mode
            )

    elif options.overFitTest:
        # --- 과적합 테스트 모드 ---
        BJDD(config).modelTraining(overFitTest=True)
    
    # (다른 옵션들도 필요에 따라 추가할 수 있습니다.)
