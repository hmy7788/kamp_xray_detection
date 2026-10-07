# 이총: 위치별 검출 진단과 배포 조건 비교

기존 데이터에서 높은 검출 성능을 얻은 다음, **같은 작은 이물 신호가 다른 위치에서도 검출되는지**, **연산량을 줄였을 때 어느 성능이 유지되는지**를 확인합니다. 제공 데이터만으로 실제 장비가 놓친 제품을 찾아냈다고 주장하지 않습니다.

완료한 실험의 [그림 포함 보고서](../../../runs/chong/06_deployment_compare/report.html)와 [수치 요약](../../../runs/chong/06_deployment_compare/summary.md)을 먼저 볼 수 있습니다.

## 이번 실행에서 확인한 것

- 새로 40 epochs 학습한 1024 기준 모델은 원본 val 369장의 이물 중심 606개를 모두 찾았고, IoU F1은 99.01%였습니다. 중심 적중과 상자 일치 성능은 다른 지표입니다.
- 원본 F1은 640 FP32에서 99.34%로 높아졌지만, 고정 합성 위치 신규 검출은 GPU 기준 341/762에서 292/762로 줄었습니다. 1024에서 성공한 119곳을 놓치고, 누락한 70곳을 새로 찾았습니다. **원본 평균 점수만으로 경량화 설정을 고르면 놓칠 수 있는 변화**입니다.
- PyTorch 없는 CPU 환경의 ONNX FP32/640은 이 PC에서 1 thread p95 57.53 ms / RSS 165.7 MiB, 4 threads p95 33.67 ms / RSS 167.2 MiB였습니다. 이 모델을 CPU 배포 기준으로 둡니다.
- 이번 Conv INT8 후보는 파일이 작아졌지만 독립 CPU 1 thread p95 104.82 ms로 느려졌고, F1도 낮아져 채택하지 않습니다. FP16도 이 GPU 실험에서 속도 이득이 없었습니다.
- 후속 ROI는 같은 CPU 640 모델을 한 번만 호출합니다. 합성 위치 신규 검출 293→327/762, p95 57.53→58.06 ms였지만 원본 F1은 99.34→99.01%이고, 기존 합성 성공 79곳을 잃고 113곳을 얻었습니다. **검증 결과에서 발견한 탐색 개선 후보**로 남깁니다.
- 통합 환경과 독립 CPU 환경은 369장 전체의 예측이 일치했습니다. 이미지 위치 복원·대조 점수·test 접근 차단 등 12개 검사가 통과했습니다.

CPU: i7-11700F, GPU: RTX 3060. 위 수치는 공장 장비의 FPS나 메모리 요구량 보장이 아닙니다. 단일 시드·단일 시간 측정 순회의 결과이며, 0.53 ms 차이를 유의한 속도 차이로 주장하지 않습니다.

## 범위와 근거

- 코드와 산출물은 `src/chong/`, `runs/chong/`에만 둡니다. 다른 팀원 영역과 공유 데이터는 수정하지 않습니다.
- 기존 설정은 YOLO26n/1024/40 epochs/batch 8/seed 42입니다. 가중치가 없어 COCO 초기 가중치부터 다시 학습합니다. 이전 실험의 수치를 새 결과로 옮겨 쓰지 않습니다.
- 원래 분할은 train 1,767 / val 369 / test 396장입니다. 이 실험은 train/val만 읽습니다.
- 데이터는 장비가 선택한 NG 영상 중심입니다. 일부 빈 라벨 영상은 제품 일부만 찍힌 사례이므로 일반 정상 제품을 대표하지 않습니다.
- '금속구'라는 재질과 실제 지름은 확인되지 않았습니다. 픽셀 크기에서 mm 검출 한계를 추정하지 않습니다.
- 과거 이물 제거 실험의 76% 잔존 검출은 제거 품질 문제로 해석을 철회했습니다. 이 실험의 출발 근거로 쓰지 않습니다.

## 실험 설계

1. **기준 모델 재학습:** 기존 하네스와 동일한 설정을 사용합니다. 학습·기본 평가는 `common/run.py`로 실행합니다.
2. **train 합성 검수:** 공식 라벨 주변에서 작고 어두운 연결 성분을 찾아 주변 배경을 추정한 뒤, 원래 픽셀과의 차이만 분리합니다. 이 신호를 크기·각도·강도 변경 없이 이동합니다. 제품이나 막대 사각형을 통째로 복사하지 않습니다.
3. **val 위치 지도:** 호기별 공식 라벨 출처 2개 촬영 묶음, 배경 6개 촬영 묶음을 고정합니다. 출처와 배경의 촬영 묶음은 겹치지 않습니다. 각 배경의 추정 제품 영역에 5×5 위치를 배치하고, 기존 라벨 근처·제품 밖·픽셀값 포화가 생기는 위치를 제외합니다.
4. **세 가지 짝 비교:** 원본 / 이물 없는 인접 패치에서 같은 방식으로 얻은 잔차를 넣은 대조 / 이물 잔차 삽입. 기존 이물은 지우지 않습니다. 검출 중심이 삽입 위치의 기준 상자 안에 들어와야 적중입니다. 다른 곳의 기존 검출은 인정하지 않습니다.
5. **동일 가중치 비교:** 1024 FP32 → 640 FP32 → 640 FP16을 순서대로 비교합니다. CPU 640 FP32에서는 PyTorch/ONNX 각각 1/4 스레드를 측정합니다. 비교 시 입력은 모두 정사각형 letterbox (`rect=False`)로 맞춥니다. 여러 모델을 잇거나 추가 정밀 검사를 하는 구성은 아닙니다.
6. **INT8 후보:** ONNX 모델의 Conv만 S8S8 QDQ 방식으로 양자화합니다. 입력/출력과 다른 연산은 FP32입니다. 학습 분할에서 호기별 16개, 총 48개 서로 다른 촬영 묶음의 영상으로 MinMax 보정하며, val/test로 보정하지 않습니다. 크기·속도·검출 손실을 모두 측정해 채택 여부를 결정합니다.
7. **후속 ROI 후보:** 처음 6단계 결과를 본 뒤 추가했습니다. 원본 영상의 큰 어두운 성분에 긴 변의 12%(최소 12픽셀) 여백을 두어 자릅니다. 영역 추정이 불안정하면 전체 영상을 사용합니다. 정답 라벨로 자르기를 정하지 않으며, 1회 추론 결과 좌표를 원본으로 복원합니다. 세 가지 짝 비교에서는 원본에서 정한 ROI를 고정합니다. 원본 정답 상자가 잘리는지도 별도로 감사합니다. 최초 비교와 달리 **검증 자료를 보고 선택한 탐색 실험**입니다.

**검출 경로 일치:** 이 버전의 원래 PyTorch 추론은 one-to-many + NMS 경로입니다. ONNX에 `nms=False`를 명시하면 다른 one-to-one 경로로 전환되어 정확도 비교가 달라집니다. 최종 변환은 `nms=True, conf=0.001, iou=0.7, max_det=50`으로 원래 경로를 맞추고 NMS를 그래프에 포함합니다. 전처리 픽셀과 원본 영상 좌표 복원도 검사합니다.

표본 선택에는 모델 예측을 사용하지 않습니다. hash 정렬과 고정 seed로 후보 순서를 정하고, 검수 단계에서도 검출 성능을 보고 표본을 고르지 않습니다. 신호 출처가 호기별 2개뿐이므로 다양한 재질·형상을 대표하는 실험은 아닙니다.

### 지표를 읽는 법

- **원본 성능:** 기존 하네스의 IoU 0.5 AP, 고정 임계값의 TP/FP/FN/F1, 중심 적중 재현율. 기본 임계값 0.42는 이전 실험의 값으로 고정하고 0.10은 민감도 확인용으로 함께 기록합니다. 새 합성 결과에 맞춰 임계값을 낮추지 않습니다.
- **합성 위치 신규 검출:** 삽입 후 적중하면서 원본과 대조에서는 해당 위치 검출이 없어야 합니다. 전체 유효 삽입 수를 분모로 둔 비율과, 원본·대조 검출이 없는 사례만의 비율을 함께 기록합니다. 이는 물리 시험편 검출률이나 현장 미탐률이 아닙니다.
- **반복 촬영 처리:** 배경 촬영 묶음별 결과와 템플릿별 결과를 따로 보존합니다. 수백 개 합성 위치를 수백 개 독립 현장 시험처럼 세어 신뢰구간을 좁히지 않습니다.
- **속도:** 배치 1, 워밍업 제외, CUDA 동기화 후 PNG 읽기·변환·전처리·모델·후처리·CPU 결과 반환까지 측정합니다. 평균뿐 아니라 p50/p95/p99를 기록합니다. 파일 읽기는 OS 캐시 영향을 받으며 카메라 수신/PLC/실제 배출장치 지연은 포함하지 않습니다.
- **자원:** 모델 파라미터/가중치 파일 크기, 프로세스 RSS, PyTorch GPU 할당 피크를 구분합니다. 현재 PC의 CPU 스레드 제한은 연산 예산 비교이며 실제 엣지 장비 재현이 아닙니다. OpenCV와 연산 간 스레드는 1로 고정하고 PyTorch/ONNX의 연산 내부 스레드만 1/4로 비교합니다. 통합 평가 프로세스의 RSS에는 PyTorch도 로드되어 있으므로 ONNX 단독 배포의 최소 메모리를 뜻하지 않습니다.

### 합성의 한계

영상은 보정 전 투과량이 아니라 가공된 회색조 영상입니다. 합성은 표시 영상의 잔차를 더하는 방법이며 X선 감쇠 법칙, 두께·산란·자동 명암 보정을 재현하지 않습니다. 주변 배경 추정도 완벽하지 않습니다. 대조 처리는 합성 흔적 영향을 살피지만 실제 현장 타당성을 입증하지는 않습니다. 결과가 모두 좋으면 그대로 기록하며, 실패를 만들기 위해 강도를 낮추거나 모양을 임의로 바꾸지 않습니다.

## Windows 실행

전용 Python 환경을 권장합니다. 이번 실행의 정확한 패키지는 `environment-lock.txt`에 있습니다. CUDA 환경에서는 먼저 PyTorch 공식 CUDA wheel을 설치하고 `requirements.txt`를 설치합니다.

```powershell
# 저장소 루트에서 실행. 원본을 변경하지 않는 train/val 보기 생성.
& ./src/chong/detectability/prepare.ps1
$env:PYTHONUTF8 = '1'
$env:KAMP_MEMBER = 'lee'

# 기존 하네스를 통한 재학습. 완료 후 best.pt가 생깁니다.
Push-Location src/chong/harness
python common/run.py configs/lee/03_yolo26n_1024_reproduce.yaml
Pop-Location

# 합성 검수 및 위치 집합 생성. 이미 있으면 덮어쓰지 않습니다.
python src/chong/detectability/build_probes.py --config src/chong/detectability/pilot.json --out runs/chong/04_probe_pilot
python src/chong/detectability/build_probes.py --config src/chong/detectability/map_val.json --out runs/chong/05_probe_map_val

# qa.html에서 모든 템플릿의 출처/추정 배경/대조/삽입을 직접 검수한 뒤 기록합니다.
python src/chong/detectability/review_probes.py --probes runs/chong/05_probe_map_val --reviewer '<검수자>' --notes '<관찰한 품질과 한계>' --accept
# 학습 종료 및 QA 승인 기록이 있어야 다음 단계가 실행됩니다.
python src/chong/detectability/export_cpu.py --training-run runs/chong/03_yolo26n_1024_reproduce --out runs/chong/06_deployment_compare/exports/fp32
python src/chong/detectability/quantize_cpu.py --onnx runs/chong/06_deployment_compare/exports/fp32/baseline_640.onnx --out runs/chong/06_deployment_compare/exports/int8
python src/chong/detectability/evaluate.py --training-run runs/chong/03_yolo26n_1024_reproduce --config src/chong/detectability/benchmark.json --probes runs/chong/05_probe_map_val --onnx runs/chong/06_deployment_compare/exports/fp32/baseline_640.onnx --int8 runs/chong/06_deployment_compare/exports/int8/baseline_640_int8.onnx --out runs/chong/06_deployment_compare
python src/chong/detectability/position_diagnostics.py --input runs/chong/06_deployment_compare --train-pilot runs/chong/04_probe_pilot

# 검증 지도에서 나온 관찰로 추가한 탐색 후보. 동일 모델 1회 추론.
python src/chong/detectability/evaluate_roi.py --comparison runs/chong/06_deployment_compare --probes runs/chong/05_probe_map_val --out runs/chong/07_roi_followup
python src/chong/detectability/evaluate_roi.py --comparison runs/chong/06_deployment_compare --probes runs/chong/05_probe_map_val --out runs/chong/07_roi_followup --backend onnx --threads 1

python -m unittest discover -s src/chong/detectability -p 'test_*.py' -v
```

`prepare.ps1`은 Windows 체크아웃의 CRLF를 별도 manifest 보기에서 LF로 바꾼 뒤 고정 SHA256과 비교합니다. 원본 manifest는 그대로 보존합니다. 보기에는 train/val 연결만 있고 test 이미지 연결은 없습니다. 체크포인트·합성 이미지·좌표 캐시는 Git에서 제외합니다. 학습이 끊긴 경우 기존 하네스의 resume 기능을 이용하며, `record.json`이 있으면 기존 설정으로 `--force`를 붙입니다.

## CPU 단독 실행

`onnx_cpu.py`는 PyTorch/Ultralytics 없이 `requirements-cpu.txt`의 패키지만으로 한 장을 처리합니다. 변환이 성공한 모델 파일과 이 실행 코드를 함께 사용할 수 있습니다. 실시간 장비 연동은 포함하지 않습니다.

```powershell
python src/chong/detectability/onnx_cpu.py --model runs/chong/06_deployment_compare/exports/fp32/baseline_640.onnx --image '<검사할 PNG 경로>' --threads 1 --threshold 0.42
```

전체 비교가 끝나면 `runs/chong/06_deployment_compare/work/standalone_inputs.json`이 생성됩니다. PyTorch가 없는 별도 가상환경에 `requirements-cpu.txt`만 설치하고 아래를 실행하면 독립 CPU 프로세스의 시간/RSS를 측정합니다. 각 측정을 순서대로 실행하고, 통합 평가나 무거운 작업과 동시에 실행하지 않습니다.

```powershell
python -m venv runs/chong/local/work/venv-cpu
$cpuPython = './runs/chong/local/work/venv-cpu/Scripts/python.exe'
& $cpuPython -m pip install -r src/chong/detectability/requirements-cpu.txt
& $cpuPython src/chong/detectability/benchmark_cpu_standalone.py --model runs/chong/06_deployment_compare/exports/fp32/baseline_640.onnx --inputs runs/chong/06_deployment_compare/work/standalone_inputs.json --out runs/chong/06_deployment_compare/standalone_fp32_t1.json --threads 1
& $cpuPython src/chong/detectability/benchmark_cpu_standalone.py --model runs/chong/06_deployment_compare/exports/fp32/baseline_640.onnx --inputs runs/chong/06_deployment_compare/work/standalone_inputs.json --out runs/chong/06_deployment_compare/standalone_fp32_t4.json --threads 4
& $cpuPython src/chong/detectability/benchmark_cpu_standalone.py --model runs/chong/06_deployment_compare/exports/int8/baseline_640_int8.onnx --inputs runs/chong/06_deployment_compare/work/standalone_inputs.json --out runs/chong/06_deployment_compare/standalone_int8_t1.json --threads 1
& $cpuPython src/chong/detectability/benchmark_cpu_standalone.py --model runs/chong/06_deployment_compare/exports/fp32/baseline_640.onnx --inputs runs/chong/06_deployment_compare/work/standalone_inputs.json --out runs/chong/07_roi_followup/standalone_roi_t1.json --threads 1 --roi

# 원래 학습/평가 환경에서, 결과 일치 검사 후 최종 보고서 생성.
python src/chong/detectability/audit_results.py --comparison runs/chong/06_deployment_compare --roi runs/chong/07_roi_followup
python src/chong/detectability/report.py --input runs/chong/06_deployment_compare
```

한 장에 ROI 후보를 적용하려면 `onnx_cpu.py` 대신 `roi_cli.py`를 같은 인수로 실행합니다. 전체 영상 기준 실행과 ROI 후보 실행 모두 모델을 한 번만 호출합니다. 이번 보고서의 `verification.json`은 최종 소스 해시와 독립 환경 예측 일치 결과를 기록합니다. 기존 `record.json`의 소스 해시는 각 실행 당시 버전을 의미합니다.

참조: [Ultralytics ONNX 변환](https://docs.ultralytics.com/integrations/onnx/), [ONNX Runtime 스레드 설정](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), [정적 양자화와 정확도 검증](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). 현재 PC의 양자화 속도 개선 여부는 측정 결과로만 판단합니다.

## 결과로 결정할 것

원본 성능·합성 위치별 검출·현재 PC의 지연시간을 한 표에서 비교하여 입력 크기와 연산 정밀도를 고릅니다. 실제 장비 사양을 모르는 상태에서는 장비 FPS를 약속하는 대신 **후보별 필요한 시간/메모리 조건**을 제시합니다. 현장 추가 검증이 가능해지면 정상 통과 영상과 실제 시험편으로 타당성을 확인해야 합니다. 그 검증은 이번 제공 데이터 실험의 결과에 포함하지 않습니다.
