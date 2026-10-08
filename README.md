# X-ray 영상 기반 완제품 이물질 탐지 — 소스코드

제6회 K-인공지능 제조데이터 분석 경진대회 · 일반국민/대학(원)생 부문 · **팀 엽총창** (정연창, 허민엽, 이총)

KAMP "X-ray 검사장비 AI 데이터셋"의 완제품 X선 영상에서 약 10px 크기의 이물질(1클래스)을 검출한다.
**최종 모델은 D-FINE-N**(파라미터 3.7M), **베이스라인은 YOLOv3-tiny**이며, 두 모델 모두 이 저장소의 같은 데이터·분할·채점 코드로 학습하고 평가한다.

## 결과 요약 (test 396장, 정답 상자 663개)

| 모델 | mAP50 | mAP50-95 | Precision | Recall | F1 | FPS (CPU 4스레드) |
|---|---|---|---|---|---|---|
| YOLOv3-tiny (베이스라인) | 0.981 | 0.558 | 0.989 | 0.989 | 0.989 | 14.0 |
| **D-FINE-N (최종 모델)** | **0.989** | **0.655** | 0.988 | 0.989 | 0.989 | 10.3 |

- 보고서 표 2-5의 값이다. 신뢰도 임계값은 모델마다 val에서 F1이 최대인 값으로 정하고(D-FINE-N 0.7114), test는 그 값으로 한 번만 채점했다.
- 근거 파일: `results/<모델>/eval_report_test.json`, 최종 모델의 test 예측 결과는 **`results/dfine_n/test_predictions.csv`**.
- 비교 모델 4종(Faster R-CNN 2종, YOLO26n, RT-DETR-l)의 결과는 `results/model_comparison.md`, 코드는 `experiments/comparison_models/`.

## 빠른 재현 (제출 가중치로 추론·채점, GPU 기준 약 5분)

```powershell
conda create -n kamp python=3.10 -y
conda activate kamp
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
.\run_all.ps1                     # Linux/macOS: bash run_all.sh
```

`run_all`은 **데이터 검사 → 추론(val, test) → 채점 → 속도 측정 → 비교표 → 제출 결과와 대조**를 차례로 실행한다.
결과는 `outputs/`에 새로 생기고, 마지막 단계에서 `results/`(제출 시점 결과)와 지표가 같은지 출력한다.
**처음부터 학습까지** 재현하려면 `.\run_all.ps1 -Train`을 쓴다(RTX 4050 Laptop 기준 약 3시간).

## 폴더 구조

```text
.
├─ README.md                 이 문서 (평가자용 안내)
├─ requirements.txt          파이썬 의존성 (버전 고정)
├─ run_all.ps1 / run_all.sh  전체 파이프라인 자동 실행
├─ configs/                  학습 설정 (dfine_n.yaml = 최종 모델, yolov3_tiny.yaml = 베이스라인)
├─ data/                     학습용 데이터 (확정 v2: 회색조 PNG 2,532장 + YOLO 라벨, 분할 고정)
├─ src/kamp_xray/            공통 코드 패키지
│   ├─ common.py             경로, 데이터 버전 확인, 정답 읽기, 공통 예측 형식
│   ├─ metrics.py            모든 모델이 함께 쓰는 채점 (AP50, mAP50-95, P/R/F1, 임계값)
│   ├─ dfine.py              최종 모델 D-FINE-N 학습·추론
│   └─ yolov3_tiny.py        베이스라인 YOLOv3-tiny 학습·추론
├─ scripts/                  실행 단계별 스크립트 (아래 "단계별 실행")
├─ third_party/yolov3/       베이스라인 모델 코드 (ultralytics YOLOv3 2020년 버전 수정본, GPL-3.0)
├─ weights/                  제출 가중치 (dfine_n/best, yolov3_tiny/best.pt) + 사전학습 가중치 자리
├─ results/                  제출 시점 결과 (보고서 수치의 근거, 읽기 전용)
│   ├─ dfine_n/              최종 모델: 평가 리포트, test 예측 결과, 속도, 학습 기록
│   ├─ yolov3_tiny/          베이스라인: 같은 구성
│   ├─ comparison_models/    비교 모델 4종의 평가 기록과 6개 모델 통합 채점
│   ├─ analysis/             오류분석·합성 이물질·전처리 검증 결과 (보고서 3장, 5장)
│   └─ model_comparison.md   모델 비교표 (보고서 표 2-5)
├─ preprocessing/            KAMP 원본 → data/ 를 만든 전처리·라벨링·분할 코드 (기록)
├─ experiments/              비교 모델 4종과 분석 실험의 원 코드 (참고용)
├─ docs/                     지표 정의, 보고서 표·그림과 파일의 대응표
└─ outputs/                  재현 실행 결과가 생기는 곳 (처음에는 없음)
```

## 1. 환경 설치

| 항목 | 검증한 버전 |
|---|---|
| OS / GPU | Windows 11 Pro, NVIDIA GeForce RTX 4050 Laptop 6GB (CPU: Intel Core i7-13620H) |
| Python | 3.10 |
| PyTorch / torchvision | 2.6.0 / 0.21.0 (CUDA 12.4) |
| transformers (D-FINE-N) | 5.19.0 |
| numpy / OpenCV | 2.2.6 / 5.0.0 |

- 위 "빠른 재현"의 설치 명령을 그대로 쓴다. GPU가 없으면 PyTorch를 `pip install torch==2.6.0 torchvision==0.21.0`으로 설치하고 `.\run_all.ps1 -Device cpu`로 실행한다.
- 새 conda 환경에 `requirements.txt`만으로 설치해 전 과정이 동작하는 것을 확인했다.
- 경로에 한글이 있어도 동작하도록 이미지는 `cv2.imdecode`로 읽고, 스크립트는 UTF-8 모드(`PYTHONUTF8=1`)로 실행된다.
- PowerShell에서 스크립트 실행이 막혀 있으면 `powershell -ExecutionPolicy Bypass -File run_all.ps1`로 실행한다.

## 2. 데이터

`data/`에 학습·평가에 쓴 확정 데이터 전체가 들어 있다. 자세한 설명은 `data/README.md`.

| | train | val | test | 합계 |
|---|---|---|---|---|
| 이미지 | 1,767 | 369 | 396 | 2,532 |
| 이물질 상자 | 3,225 | 606 | 663 | 4,494 |
| 공식 라벨 / 팀 라벨 이미지 | 356 / 1,411 | 59 / 310 | 85 / 311 | 500 / 2,032 |

- **형식**: 회색조 PNG, YOLO 라벨 `0 cx cy w h`(0~1 비율). `images/`와 `labels/`는 형제 폴더다.
- **버전 확인**: `manifest.csv`의 sha256이 `1942bf34...623c623`이어야 한다. 모든 스크립트가 실행할 때마다 확인하고, 다르면 멈춘다.
- **분할은 고정**되어 있다. 60초 이내 연속 촬영 묶음 547개 단위로 무작위 분할(seed 42)했으며 다시 만들지 않는다.
- **전처리**: KAMP 원본 BMP에서 중복 제거, 장비가 그린 색상 상자 제거(주변 5×5 회색 평균으로 메움), 회색조 PNG 변환을 거쳤다.
  코드는 `preprocessing/`에 있고, 원본 BMP에 다시 적용하면 `data/`와 픽셀 단위로 같은지 검증할 수 있다.

```powershell
python scripts/verify_preprocess.py --raw "<KAMP 원본>/dataset/test1/yolov3"   # 2,532장 모두 일치해야 함
```

## 3. 단계별 실행

`run_all`이 아래를 순서대로 부른다. 모두 저장소 루트에서 실행한다.

| 단계 | 명령 | 결과 |
|---|---|---|
| 데이터 검사 | `python scripts/check_data.py` | `outputs/data_check.json` |
| (학습 시) 사전학습 가중치 | `python scripts/download_pretrained.py` | Hugging Face 캐시, `weights/pretrained/yolov3-tiny.pt` |
| (학습 시) 학습 | `python scripts/train.py --model dfine_n` | `outputs/dfine_n/train/` |
| 추론 | `python scripts/predict.py --model dfine_n --split val` (test도 같게) | `outputs/dfine_n/preds_val.json` |
| 채점 | `python scripts/evaluate.py --model dfine_n --split val` → `--split test` | `eval_report_{val,test}.json`, `test_predictions.csv` |
| 속도 | `python scripts/speed.py --model dfine_n` | `outputs/dfine_n/speed.json` |
| 비교표 | `python scripts/compare.py` | `outputs/model_comparison.md` |
| 제출 결과와 대조 | `python scripts/compare.py --check` | 화면 출력 |

베이스라인은 `--model yolov3_tiny`로 같은 명령을 쓴다. 직접 학습한 가중치로 추론하려면 `--weights outputs/<model>/train/weights/...`를 붙인다.

### 학습 설정 요약 (`configs/`)

| | D-FINE-N (최종) | YOLOv3-tiny (베이스라인) |
|---|---|---|
| 시작 가중치 | COCO 사전학습 `ustc-community/dfine-nano-coco` | COCO 사전학습 yolov3-tiny |
| 입력 | 640×640 (늘림) | 640 (letterbox) |
| epoch | 최대 50, patience 10 (제출 모델: 43에서 종료, 최적 33) | 100 |
| batch | 16 | 16 |
| 증강 | 밝기 ±30%, 좌우·상하 뒤집기, 이동 0.1, 크기 ±0.2 | yolov3 기본값 |
| 체크포인트 선택 | val 0.1×AP50 + 0.9×AP50-95 | val 0.99×mAP50 + 0.01×R |
| 학습 시간 (RTX 4050 Laptop) | 약 98분 | 약 59분 |

약 10px 이물질이 더 작아지지 않도록 mosaic과 큰 축소 증강은 쓰지 않았고, 회색조 영상이라 색상 증강도 쓰지 않았다.

## 4. 평가 규칙 (`src/kamp_xray/metrics.py`)

- **매칭**: 이미지마다 신뢰도가 높은 검출부터, IoU 0.5 이상이며 아직 짝이 없는 정답 중 IoU가 가장 큰 것과 짝짓는다. 정답 하나에는 검출 하나만 대응한다.
- **AP**: 신뢰도 0.001 이상의 모든 검출로 정밀도-재현율 곡선을 만들고 전 구간 보간(VOC)으로 면적을 구한다. mAP50-95는 IoU 0.50~0.95(0.05 간격) AP의 평균이다.
- **임계값**: val에서 F1이 최대인 신뢰도로 정하고 test에는 그 값을 그대로 쓴다. test는 모델 선택과 임계값 결정에 쓰지 않았다.
- **보조 지표**: 중심 거리 1~5px 매칭(상자 크기와 무관한 위치 정확도), 이미지 단위 판정(영상 내 최고 신뢰도 ≥ 임계값이면 불합격), 라벨 출처·호기·해상도·월별 분해.
- **속도**: 배치 1, CPU 4스레드, val 100장, 예열 10장, 3회 반복의 중앙값. 전처리·추론·후처리를 포함하고 파일 읽기는 제외한다.

## 5. 제출 결과물 (`results/`)

| 파일 | 내용 |
|---|---|
| `results/dfine_n/test_predictions.csv` | **최종 모델의 test 예측 결과** (임계값 이상 상자: image_id, x1, y1, x2, y2, score, 원본 픽셀 좌표) |
| `results/dfine_n/test_image_decisions.csv` | test 이미지별 최고 신뢰도와 불합격·합격 판정 |
| `results/dfine_n/preds_test.json` | 신뢰도 0.001 이상 전체 검출 (채점 입력) |
| `results/<모델>/eval_report_{val,test}.json` | 임계값, 전체 지표, IoU별 AP, 중심 거리 지표, 조건별 분해 |
| `results/<모델>/speed.json` | 처리 속도 |
| `results/<모델>/original_run/` | 제출 가중치를 만든 원 학습 실행의 설정·로그·학습 곡선 |
| `results/model_comparison.md` | 6개 모델 비교표 |

보고서의 표·그림과 파일의 대응은 `docs/report_mapping.md`에 있다.

## 6. 재현할 때 알아 둘 점

- **제출 가중치로 추론·채점하면 같은 수치가 나온다.** 다시 학습하면 GPU 연산의 비결정성 때문에 수치가 조금 달라질 수 있다.
- 속도는 하드웨어와 전원 상태에 크게 좌우된다. 보고서 수치는 노트북 AC 전원, 전원 모드 "최고 성능", 고성능 코어 고정(`--affinity 0,2,4,6`)에서 쟀다.
- 데이터에는 기존 검사 장비가 불량으로 판정한 영상만 있고 정상 제품 영상은 없다. 그래서 정상품 오검출률은 검증 범위 밖이다(보고서 1.1절, 4.1절).
- 팀 라벨은 클릭한 점을 중심으로 고정 크기 상자를 만든 것이라 상자 크기 정보가 없다. IoU 기준 오류는 대부분 공식 라벨과의 크기 차이에서 생긴다(보고서 3.2절).

## 7. 출처와 라이선스

- 데이터: KAMP(한국인공지능제조플랫폼) "X-ray 검사장비 AI 데이터셋". 대회 제출 목적으로만 포함했다.
- D-FINE-N: Hugging Face `transformers`의 `DFineForObjectDetection`, 사전학습 가중치 `ustc-community/dfine-nano-coco`(Apache-2.0).
- YOLOv3-tiny: [ultralytics/yolov3](https://github.com/ultralytics/yolov3) 2020년 버전 수정본(GPL-3.0). 수정 내역은 `third_party/yolov3/README.md`.
- 사전학습 YOLOv3-tiny 가중치: Darknet 공식 COCO 가중치(pjreddie.com)를 변환해 쓴다(`scripts/download_pretrained.py`).
