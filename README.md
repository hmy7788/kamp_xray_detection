# X-ray 검사장비 이물질 검출 (YOLOv3)

KAMP(K-인공지능 제조 플랫폼) **X-ray 검사장비 AI 데이터셋**으로 완제품 X선 영상의 이물질(`defect`)을 검출하는 프로젝트입니다.
2026년 제6회 K-인공지능 제조데이터 분석 경진대회 과제(**영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석**)를 배경으로 합니다.

- 마감: **2026-10-08 23:59** (서면평가 결과 10-26, 발표평가 10-30)
- 평가(100점): 데이터 이해·진단 15 / **모델 개발 40**(베이스라인 포함 2개 이상 비교) / 영향요인·오류분석 15 / 현장 활용 10 / 창의성 10 / 코드·재현성 10 → [docs/evaluation.md](docs/evaluation.md)
- **제출물(코드, 문서, 커밋 포함)에 소속·로고 등 식별 정보를 넣지 않습니다.** 성명, 팀명만 허용됩니다.

## 목차

1. [빠른 시작](#1-빠른-시작)
2. [요구 사항](#2-요구-사항)
3. [환경 설치](#3-환경-설치-conda)
4. [데이터 준비](#4-데이터-준비)
5. [사전학습 가중치 받기](#5-사전학습-가중치-받기)
6. [실행 방법](#6-실행-방법)
7. [프로젝트 구조](#7-프로젝트-구조)
8. [협업 가이드](#8-협업-가이드)
9. [현재 상태와 할 일](#9-현재-상태와-할-일)
10. [자주 겪는 문제](#10-자주-겪는-문제)
11. [라이선스 및 출처](#11-라이선스-및-출처)

## 1. 빠른 시작

처음 받는 사람은 아래 순서로 진행합니다. 각 단계의 자세한 설명은 뒤에 있습니다.

```text
① conda 환경 KAMP 만들기 + 라이브러리 설치        (3장)
② KAMP 데이터를 data/subsets/n500 에 배치        (4장)   → python scripts/check_data.py 로 확인
③ 사전학습 가중치를 weights/pretrained 에 저장    (5장)
④ python scripts/run_experiment.py ... 로 실험 1건 실행   (6장)
```

## 2. 요구 사항

| 항목 | 내용 |
|---|---|
| OS | **Windows 11에서만 검증**했습니다. Linux/macOS는 시도하지 않았습니다. |
| Python | 3.10 (conda 환경) |
| GPU | 권장. 개발 환경은 NVIDIA RTX 4050 Laptop **6GB**. GPU가 없어도 `--device cpu`로 실행되지만 매우 느립니다. |
| 디스크 | 가중치와 데이터를 합쳐 약 3GB 여유 |
| 도구 | [Anaconda 또는 Miniconda](https://docs.conda.io/en/latest/miniconda.html), Git |

검증한 라이브러리 버전: torch 2.6.0+cu124, torchvision 0.21.0, numpy 2.2.6, opencv-python 5.0.0.93, matplotlib 3.10.9, scipy 1.15.3, tensorboard 2.21.0.

## 3. 환경 설치 (conda)

**Anaconda Prompt**(또는 PowerShell)에서 실행합니다.

### 3-1. 저장소 받기

```bash
git clone <저장소 URL> KAMP
cd KAMP
```

### 3-2. 가상환경 만들기

```bash
conda create -n KAMP python=3.10 -y
conda activate KAMP
```

이후 모든 명령은 `KAMP` 환경이 활성화된 상태(프롬프트 앞에 `(KAMP)` 표시)에서 실행합니다.

### 3-3. PyTorch 설치

GPU 환경에 맞는 것 **하나만** 고릅니다. (`nvidia-smi`로 드라이버를 확인하세요. 개발 환경은 CUDA 12.4용 휠을 사용했습니다.)

```bash
# (A) NVIDIA GPU, CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# (B) GPU 없음 (CPU 전용)
pip install torch==2.6.0 torchvision==0.21.0
```

### 3-4. 나머지 라이브러리 설치

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name kamp --display-name KAMP
```

루트의 `requirements.txt`는 검증한 버전으로 고정되어 있고 Jupyter(`jupyter`, `ipykernel`)도 포함합니다. 3-3에서 PyTorch를 먼저 설치했다면 `torch`/`torchvision`은 "이미 설치됨"으로 건너뜁니다. (`third_party/yolov3/requirements.txt`는 원본 yolov3의 느슨한 버전 조건(`>=`)이라 재현에는 위 파일을 쓰세요.)

### 3-5. 설치 확인

```bash
python -c "import torch, cv2, numpy; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), cv2.__version__, numpy.__version__)"
```

`CUDA: True`가 나오면 GPU를 쓸 수 있는 상태입니다. GPU가 있는데 `False`면 PyTorch를 (A)로 설치했는지, 드라이버가 최신인지 확인하세요.

### 3-6. 한글 경로용 환경 변수 (중요)

프로젝트 경로에 한글이 들어갈 수 있고(예: Windows 사용자 폴더), 코드가 설정 파일을 열 때 기본 인코딩(cp949) 때문에 실패할 수 있습니다. **터미널에서 스크립트를 직접 실행할 때마다** 아래를 먼저 설정하세요. (`run_experiment.py`는 자식 프로세스에 자동으로 넣어 주고, 노트북도 자동 설정합니다.)

```bash
# Anaconda Prompt(cmd)
set PYTHONUTF8=1

# PowerShell
$env:PYTHONUTF8 = "1"
```

## 4. 데이터 준비

**이 저장소에는 이미지와 라벨이 들어 있지 않습니다**(KAMP 데이터의 재배포 조건 확인 전이라 `.gitignore`로 제외). 각자 KAMP에서 데이터를 받아 배치해야 합니다. 저장소에는 파일명과 메타데이터만 있습니다.

| 저장소에 있는 것 | 설명 |
|---|---|
| `data/manifest.csv` | 라벨된 500장의 이미지별 실제 호기, 해상도, bbox 수 |
| `data/splits/split.csv` | train/val/test 분할 기록 |
| `data/classes.names` | 클래스 이름 (`defect` 1개) |

### 4-1. 필요한 데이터

KAMP "X-ray 검사장비 AI 데이터셋"의 **라벨된 이미지 500장과 라벨**입니다. 이미지가 두 곳에 나뉘어 있습니다.

- 400장: `라벨링 6종 세트/images 400/` (확장자 jpg, 실제 내용은 BMP)
- 100장: 원본 폴더 `test1/yolov3/X선이물검출기(06.23_09.22)/` 안의 bmp (라벨 이름과 같은 파일명으로 찾음)
- 라벨 500개: `라벨링 6종 세트/labels/` (YOLO txt 한 폴더에 모두 있음)

다음처럼 배치합니다. 파일명은 `data/manifest.csv`와 같아야 합니다.

```text
data/subsets/n500/
├─ images/   001_20200622_203305(8).jpg ...   (500장. 400장은 확장자만 jpg이고 실제 내용은 BMP, 추가 100장은 .bmp)
└─ labels/   001_20200622_203305(8).txt ...   (이미지와 같은 이름, YOLO 포맷)
```

- 라벨 500개를 모두 `labels/`에 넣고, 이미지 500장의 이름과 1:1로 맞아야 합니다. (`scripts/check_data.py`가 확인)
- 라벨 포맷: 한 줄에 `class x_center y_center width height`(0~1 정규화). 클래스는 0(`defect`) 하나입니다.

### 4-2. 배치 확인

```bash
python scripts/check_data.py
```

`OK: 데이터 배치가 올바릅니다.`가 나오면 됩니다. 누락/초과 파일과 bbox 합계(1,147)를 점검해 문제를 알려 줍니다.

### 4-3. 데이터에 대해 알아둘 것

- 라벨된 500장은 **1호기 156 / 2호기 177 / 3호기 167장**이고 호기마다 해상도가 다릅니다(352x332 / 316x332 / 576x444).
- **파일명 접두(001/002)는 호기가 아닙니다.** 002 = 1·2호기, 001 = 3호기입니다. 호기는 `data/manifest.csv`의 `unit` 열을 쓰세요.
- 결함은 이미지의 1~5%인 **매우 작은 객체**이고, 모든 이미지가 NG 판정이라 **정상(결함 없음) 이미지는 없습니다**.
- 이전 서브셋(n015~n400)은 서로 포함 관계이고 `n400`이 최대입니다. 현재는 추가 100장을 합친 `n500`을 씁니다.

자세한 수치는 [docs/dataset.md](docs/dataset.md), 원본 폴더 조사는 [docs/original_folder_inventory.md](docs/original_folder_inventory.md)를 참고하세요.

### 4-4. train/val/test 분할

호기별로 층화하고, 같은 호기에서 60초 이내 연속 촬영된 이미지는 같은 분할에 넣어 누수를 막습니다(seed 42). 결과는 train 400 / val 51 / test 49장입니다.

```bash
python scripts/make_split.py
```

`data/splits/`에 `train.txt`, `val.txt`, `test.txt`, `val.data`, `test.data`가 생깁니다. 이 파일들에는 **본인 PC의 절대경로**가 들어 있어 Git에 올리지 않으며(`.gitignore`), 실행할 때마다 다시 만들어집니다. 분할 자체는 시드가 고정되어 모든 사람이 같은 결과를 얻어야 합니다(`git diff data/splits/split.csv`에 변경이 없어야 함).

> 분할을 바꾸면 이전 실험과 수치를 비교할 수 없습니다. 바꾸기 전에 팀과 상의하세요.

## 5. 사전학습 가중치 받기

COCO로 사전학습된 가중치에서 시작합니다. (`weights/legacy`의 가중치는 쓰지 마세요. 어떤 이미지로 학습됐는지 알 수 없어 평가에 누수가 생길 수 있습니다.)

```bash
mkdir weights\pretrained        # Linux/macOS: mkdir -p weights/pretrained
curl.exe -L -o weights/pretrained/yolov3-spp-ultralytics.pt https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-spp-ultralytics.pt
curl.exe -L -o weights/pretrained/yolov3-tiny.pt https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-tiny.pt
```

- `yolov3-spp-ultralytics.pt` 약 252MB, `yolov3-tiny.pt` 약 35MB입니다.
- 가중치 파일은 Git에 올리지 않습니다(`weights/*`, `*.pt`는 `.gitignore`).
- 다른 URL(`v9.0` 등)은 404였고 `v8` 경로만 확인했습니다.

## 6. 실행 방법

모든 명령은 **저장소 루트**(`KAMP/`)에서, `KAMP` 환경에서, `PYTHONUTF8=1`을 설정한 뒤 실행합니다.

### 6-1. 실험 1건 실행 (권장)

학습 → val/test 평가 → 결과 저장까지 한 번에 실행합니다. 분할도 자동으로 다시 만듭니다.

```bash
# 베이스라인: YOLOv3-SPP (GPU 6GB 기준 batch 3 권장. batch 4는 메모리 한계에 가까움)
python scripts/run_experiment.py --name 01_yolov3spp_coco --cfg yolov3-spp.cfg --weights weights/pretrained/yolov3-spp-ultralytics.pt --epochs 100 --batch-size 3

# 비교 모델: YOLOv3-tiny
python scripts/run_experiment.py --name 02_yolov3tiny_coco --cfg yolov3-tiny.cfg --weights weights/pretrained/yolov3-tiny.pt --epochs 100 --batch-size 8
```

| 인자 | 의미 |
|---|---|
| `--name` | 실험 이름(폴더명). **이미 있는 이름이면 덮어쓰지 않고 종료**합니다. |
| `--cfg` | 모델 정의. `yolov3-spp.cfg`, `yolov3-tiny.cfg` (`third_party/yolov3/`, 1클래스로 설정됨) |
| `--weights` | 시작 가중치 (저장소 루트 기준 상대경로 또는 절대경로) |
| `--epochs` | 학습 epoch. 체크포인트가 이미 N epoch 학습된 것이면 거기서 **추가로 N epoch**입니다. COCO 사전학습본은 0부터 시작합니다. |
| `--batch-size` | 6GB GPU에서 SPP는 4 이하, tiny는 8 정도. 메모리 부족이면 줄이세요. |
| `--img-size` | 학습 해상도 `[최소 최대]`, 기본 `320 640` |
| `--device` | `cpu` 또는 GPU 번호. 비워 두면 GPU 자동 사용 |
| `--extra ...` | `train.py`에 그대로 넘길 추가 인자 (**맨 뒤에** 둠) |

**진행 상황 보기 (백그라운드로 돌려도 확인 가능)**

실험은 단계와 epoch마다 한 줄씩 출력하고(`progress.log`), 10초마다 `status.json`을 갱신합니다. 별도 터미널에서 요약을 볼 수 있습니다.

```bash
python scripts/watch_experiment.py                   # 가장 최근 실험
python scripts/watch_experiment.py 01_yolov3spp_coco # 이름 지정
python scripts/watch_experiment.py --follow          # 10초마다 갱신 (Ctrl+C로 종료)
```

단계, epoch, 경과·남은 시간, 최근 val 지표, 최고 F1을 보여 주고, **5분 넘게 갱신이 없으면 "멈춘 것 같음"** 경고를 냅니다. 창 없이 백그라운드로 실행하려면 PowerShell에서 `Start-Process`를 쓰면 됩니다. 실행 중에는 Windows 절전 진입을 스크립트가 막습니다(시스템 설정은 바꾸지 않음).

**결과 위치** `outputs/runs/<실험명>/`

| 파일 | 내용 |
|---|---|
| `weights/best.pt`, `last.pt` | 가중치 (Git 제외) |
| `results.txt`, `results.png` | epoch별 손실과 val 지표, 곡선 |
| `train.log`, `val_eval.log`, `test_eval.log` | 로그 |
| `progress.log`, `status.json` | 진행 상황 (Git 제외) |
| `metrics.json` | val/test의 P, R, mAP@0.5, F1 |
| `config.json` | 실행 설정 |

- `best.pt`는 val에서 `0.1·mAP + 0.9·F1`이 가장 높은 epoch의 가중치입니다. 평가에는 `best.pt`가 있으면 그것을 씁니다.
- **모델/임계값 선택은 val로만 하고, test는 최종 보고용으로 한 번만 봅니다.**
- epoch가 `--epochs`보다 적게 끝나면 불완전 종료로 처리하고 실패 메시지를 냅니다.
- 소요 시간: 개발 PC에서 epoch당 약 1분(tiny)~50초(SPP)로 관찰했고 100 epoch에 1시간 이상 걸릴 수 있습니다(`persistent_workers` 적용 후 속도는 아직 측정하지 못함).
- **긴 학습 전에 PC의 절전/최대 절전을 끄세요.** 절전에 들어가며 학습이 멈춘 사례가 있습니다([docs/troubleshooting.md](docs/troubleshooting.md) 19).

**평가 (권장)**: 학습이 끝난 실험을 val에서 정한 신뢰도 임계값으로 평가합니다.

```bash
python scripts/evaluate.py 01_yolov3spp_coco
```

val에서 bbox F1이 최대인 임계값을 고르고 같은 값을 test에 적용해 P/R/F1, TP/FP/FN, AP@0.5, 이미지 검출률, 호기별 F1을 출력하고 `eval_report_img<해상도>.json`을 저장합니다. 검출 결과는 `preds_*.json`에 캐시됩니다(정답 좌표 포함이라 Git 제외).

### 6-2. 학습/추론/평가를 따로 실행

`third_party/yolov3`에서 실행합니다. 데이터·가중치는 **절대경로**로 넘기세요(아래 `<KAMP>`는 저장소 루트의 절대경로).

```bash
cd third_party/yolov3

# 학습 (val.data로 매 epoch 검증)
python train.py --epochs 100 --weights <KAMP>/weights/pretrained/yolov3-tiny.pt --batch-size 8 --cfg yolov3-tiny.cfg --data <KAMP>/data/splits/val.data

# 추론 (결과 이미지는 --output 폴더에 저장)
python detect.py --weights <가중치.pt> --source <이미지 폴더 또는 파일> --cfg yolov3-tiny.cfg --names <KAMP>/data/classes.names --output <결과 폴더>

# 평가 (test 분할)
python test.py --cfg yolov3-tiny.cfg --batch-size 8 --data <KAMP>/data/splits/test.data --weights <가중치.pt>
```

- `train.py`의 `--data`/`--cfg` 기본값은 COCO용이라 **항상 명시**해야 합니다.
- 학습 결과는 기본적으로 `third_party/yolov3/weights/last.pt`에 저장되어 **덮어써집니다**(`--nosave`를 써도 마지막 epoch는 저장됨). 실험 폴더에 저장하려면 환경 변수 `YOLO_SAVE_DIR=<폴더>`를 지정하거나 `run_experiment.py`를 쓰세요.

### 6-3. 노트북

`notebooks/yolov3_refactored.ipynb`는 원본 실습의 흐름(데이터 확인 → 분할 → 학습 → 추론 → 평가)을 경로 하드코딩 없이 재구성한 것입니다. 커널을 **KAMP**로 선택해 위에서부터 실행합니다.

- 원본 폴더(`4. X-ray 검사장비 AI 데이터셋/`)가 없으면 ②의 데이터 확인·재정리 셀만 "원본 폴더 없음"을 출력하고 건너뜁니다.
- 그 뒤 셀들은 실습 세트 `data/samples/{images,labels}`(실습용 15장)를 사용하는데, 이 폴더는 Git에 없습니다. 없으면 오류가 납니다. 노트북을 돌려 보려면 `data/subsets/n400`에서 이미지·라벨 15쌍을 `data/samples/`로 복사해 쓰세요(수치는 달라집니다).
- 실험의 기준 경로는 6-1의 스크립트이고, 노트북은 실습 흐름 확인용입니다.

## 7. 프로젝트 구조

```text
KAMP/
├─ README.md  CLAUDE.md          # 협업 안내 / AI 코딩 도구용 저장소 가이드
├─ configs/                      # 실험 설정 (예정)
├─ data/                         # classes.names, manifest.csv, splits/ (이미지·라벨은 각자 배치)
├─ src/                          # 재사용 코드: data / models / evaluation (예정)
├─ scripts/
│  ├─ run_experiment.py          # 학습 → 평가 → 결과 저장 (실험 1건, 진행 상황 출력)
│  ├─ watch_experiment.py        # 실험 진행 상황 보기 (멈춤 경고 포함)
│  ├─ evaluate.py                # val에서 임계값을 고정해 bbox/이미지 단위 P/R/F1 계산
│  ├─ reeval_experiment.py       # 끝난 실험을 다른 해상도로 재평가
│  ├─ make_split.py              # 호기 층화 + 묶음 단위 8:1:1 분할
│  ├─ build_manifest.py          # 원본 폴더가 있을 때만: 이미지별 호기 판별
│  └─ check_data.py              # 데이터 배치 점검
├─ notebooks/                    # 리팩토링 노트북, 원본 노트북
├─ third_party/
│  ├─ yolov3/                    # 호환성 수정본 YOLOv3 (train/test/detect.py)
│  └─ OpenLabeling/              # 라벨링 도구
├─ weights/                      # 사전학습 가중치 (Git 제외)
├─ outputs/runs/<실험명>/        # 실험 산출물 (가중치 제외하고 Git 포함)
├─ reports/  submission/         # 보고서용 그림·표 / 제출물
└─ docs/                         # 문서 (아래)
```

| 문서 | 내용 |
|---|---|
| [docs/dataset.md](docs/dataset.md) | 데이터 현황, 호기별 통계, 분할 |
| [docs/original_folder_inventory.md](docs/original_folder_inventory.md) | 원본 폴더 전수 조사 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | 환경/코드 호환 문제와 해결 기록 |
| [docs/experiment_results.md](docs/experiment_results.md) | 실험 설정과 결과 |
| [docs/evaluation.md](docs/evaluation.md) | 평가 기준과 현재 상태 |
| [docs/strategy.md](docs/strategy.md) | 전략 검토 |
| [docs/structure.md](docs/structure.md) | 디렉터리 구조 설계 |

## 8. 협업 가이드

### 브랜치와 커밋

- `main`에 직접 작업하지 않고 작업 브랜치를 만듭니다. 예: `feat/cv-5fold`, `fix/eval-threshold`, `docs/report-draft`.
- 커밋은 **작업 단위로 쪼개고**, 메시지는 `종류: 한글 요약` 형식을 씁니다. 종류는 `feat`, `fix`, `docs`, `chore`를 사용합니다. (예: `feat: 묶음 기준 5-fold 교차검증 스크립트 추가`)
- 변경 이유를 본문에 적습니다.

### 실험 규칙

- 실험 하나 = `outputs/runs/<번호>_<모델>_<설정>/` 폴더 하나. 예: `03_yolov3tiny_img640`. **공용 `last.pt`에 덮어쓰지 않습니다.**
- 실험은 `scripts/run_experiment.py`로 실행해 결과 형식을 맞춥니다.
- 비교 실험은 **같은 분할(`data/splits`)과 같은 지표**를 씁니다. 분할/시드/하이퍼파라미터를 바꿨다면 `docs/experiment_results.md`에 이유를 적습니다.
- 실험을 끝낸 뒤 `docs/experiment_results.md` 표를 채웁니다. **미완료·중단된 실험은 최종 결과로 인용하지 않고** 폴더 이름을 `_incomplete_...`로 바꿔 구분합니다.
- 새로 겪은 문제와 해결은 `docs/troubleshooting.md`에 추가합니다.

### 올리면 안 되는 것

| 항목 | 이유 |
|---|---|
| 가중치 `*.pt`, `weights/` | 파일당 35~500MB로 GitHub 100MB 제한 초과 |
| 이미지·라벨(`data/subsets` 등), 원본 데이터 폴더 | KAMP 데이터 재배포 조건 미확인 |
| `data/splits/*.txt`, `*.data` | 개인 PC 절대경로 포함 |
| 개인 이름·소속·이메일·PC 경로가 든 파일 | **대회 규칙: 식별 정보 금지** |

- 실험 로그와 `config.json`에는 로컬 절대경로가 찍힙니다. **커밋 전에 `<KAMP>`로 치환**하고 아래 명령으로 확인하세요. 출력이 없어야 합니다.

  ```bash
  git grep --untracked -nI "OneDrive\|Users[\\/]" -- outputs docs notebooks scripts
  ```

  이미 올라간 로그는 같은 방식으로 치환되어 있습니다.
- 노트북은 **출력(output)을 지우고** 커밋합니다 (경로와 개인정보가 출력에 남을 수 있음).
- `.gitignore`가 대부분 막아 주지만, 커밋 전 `git status`로 확인하세요.

### 코드 수정 시 주의

- `third_party/yolov3`는 2020년 버전을 최신 PyTorch/numpy/Windows에서 돌리도록 **이미 수정**했습니다(`torch.load`의 `weights_only=False`, `np.int` 제거, 추가 epoch 계산, 한글 경로용 `imread`/`imwrite` 등). 원본으로 되돌리지 마세요. 수정 목록은 [docs/troubleshooting.md](docs/troubleshooting.md)와 `CLAUDE.md`에 있습니다.
- 재사용할 코드는 `third_party`가 아니라 `src/`에 작성합니다.

## 9. 현재 상태와 할 일

> 2026-10-02 기준. **⚠ 지금까지의 베이스라인 수치(test F1 0.95~0.97)는 결함 검출 성능이 아닙니다.** 모든 이미지에 장비가 그린 색 박스(표시)가 찍혀 있고, 두 모델은 이를 보고 맞혔습니다. 표시를 지운 입력에서는 두 모델 모두 TP 0입니다([docs/experiment_results.md](docs/experiment_results.md), [docs/dataset.md](docs/dataset.md) 8절). **표시를 제거한 이미지로 다시 학습·평가해야 합니다.**

| 우선순위 | 할 일 | 평가 항목 | 담당 |
|---|---|---|---|
| 1 | **장비 표시(색 박스) 제거 이미지 만들기 → 두 베이스라인 재학습·재평가** (기존 결과는 표시를 본 것이라 무효) | 모델 개발(40) | |
| 2 | ~~이미지 단위·bbox 단위 평가 스크립트(val 임계값 고정)~~ (완료: `scripts/evaluate.py`) | 모델 개발(40) | |
| 3 | 호기·크기·날짜별 FN/FP 집계와 시각화 (**AI 미탐지 조건 분석**) | 오류분석(15) | |
| 4 | 묶음 기준 5-fold 교차검증 | 모델 개발(40) | |
| 5 | 개선 실험(해상도 확대, 타일링, 의사라벨, 앙상블 등 — 효과는 미검증) | 모델·창의성 | |
| 6 | 한 번에 도는 실행 스크립트, 테스트 예측결과 CSV (`requirements.txt`는 완료, `environment.yml`은 선택) | 재현성(10) | |
| 7 | 보고서(KAMP 제공 양식), 발표자료 | 전 항목 | |

**먼저 확인할 것**
- KAMP의 **공식 데이터(약 1.8GB), 결과 보고서 양식, 예측결과 파일 형식**이 이 저장소의 데이터와 같은지 확인 필요(현재 확인하지 못함).
- 공식 지표가 이미지 단위인지 bbox 단위인지 불명확합니다. 두 가지를 모두 계산해 두는 것이 안전합니다.
- 원본 데이터에는 정상 이미지가 없어 이미지 단위 오검출(FP) 평가에는 별도 정상 데이터가 필요합니다.

## 10. 자주 겪는 문제

| 증상 | 원인과 해결 |
|---|---|
| `UnicodeDecodeError: 'cp949'...` | `PYTHONUTF8=1`을 설정하지 않음 (3-6) |
| `Image Not Found <경로>` (파일은 있음) | 한글 경로를 `cv2.imread`가 읽지 못함. `third_party/yolov3`의 `imread` 헬퍼를 쓰는 코드인지 확인 |
| `CUDA: False` | CPU용 PyTorch를 설치했거나 드라이버 문제 (3-3, 3-5) |
| CUDA out of memory | `--batch-size`를 줄임. SPP는 6GB에서 batch 4가 한계에 가까움 |
| `이미 존재하는 실험 폴더입니다` | `--name`을 새 이름으로 바꾸거나 기존 폴더 이름을 `_incomplete_...`로 변경 |
| `학습이 N/100 epoch에서 끝났습니다(불완전)` | 학습이 중간에 종료됨. `train.log` 확인, 절전 해제 후 재실행 |
| 학습이 멈춘 채 프로세스만 남음 | 절전 추정. 프로세스를 종료하고 재실행 ([troubleshooting.md](docs/troubleshooting.md) 19) |
| `check_data.py`가 문제를 보고 | 4장대로 이미지·라벨을 다시 배치 |
| 폴더 이름 변경 시 "다른 프로세스가 사용 중" | 학습 프로세스가 남아 있는지 확인 후 종료 |

그 밖의 오류는 [docs/troubleshooting.md](docs/troubleshooting.md)를 먼저 확인하세요. 새로운 문제라면 해결 후 그곳에 추가해 주세요.

## 11. 라이선스 및 출처

- 학습 코드: [ultralytics/yolov3](https://github.com/ultralytics/yolov3) (GPL-3.0) 2020년 버전 기반으로 수정해 `third_party/yolov3`에 포함. 현재 `third_party/yolov3`에 LICENSE 파일은 없습니다(공개 저장소로 올리기 전에 원본 라이선스 파일을 추가해야 합니다).
- 라벨링 도구: [OpenLabeling](https://github.com/Cartucho/OpenLabeling) (Apache-2.0), `third_party/OpenLabeling/LICENSE` 포함
- 데이터셋: KAMP (https://www.kamp-ai.kr). 이용 약관은 각자 확인하세요.
