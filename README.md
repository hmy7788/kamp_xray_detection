# X-ray 검사장비 이물질 검출 (YOLOv3-SPP)

KAMP(K-인공지능 제조 플랫폼) **X-ray 검사장비 AI 데이터셋**을 이용해 완제품 X선 영상에서 이물질(`defect`)을 검출하는 프로젝트입니다. 2026년 제6회 K-인공지능 제조데이터 분석 경진대회 과제(영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석)를 배경으로 합니다.

## 프로젝트 구성

```
KAMP/
├─ README.md  CLAUDE.md
├─ configs/                # 실험 설정 (예정)
├─ data/                   # 정리된 데이터: samples, subsets(n015~n400), splits(train/val/test), classes.names
├─ src/                    # 재사용 코드: data / models / evaluation (예정)
├─ scripts/                # 실행 진입점 (make_split.py)
├─ notebooks/              # yolov3_refactored.ipynb, 원본 노트북
├─ third_party/            # yolov3 (호환성 수정본), OpenLabeling
├─ weights/legacy/         # 기존 가중치 사본
├─ outputs/runs/           # 실험별 산출물
├─ reports/  submission/   # 보고서 자료, 제출물
├─ docs/                   # dataset / troubleshooting / experiment_results / evaluation / structure
└─ 4. X-ray 검사장비 AI 데이터셋/   # 원본 실습 폴더 (출처로 보존, 수정하지 않음)
```

상세 설명과 평가 항목별 담당 위치는 [docs/structure.md](docs/structure.md)를 참고하세요.

## 환경 설정

Anaconda 기준입니다.

```bash
conda create -n KAMP python=3.10
conda activate KAMP
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # CPU만 쓰면 기본 pip 인덱스
pip install -r third_party/yolov3/requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name kamp --display-name KAMP
```

검증한 환경: Windows 11, Python 3.10, PyTorch 2.6.0+cu124, numpy 2.2, RTX 4050 Laptop(6GB).

> 프로젝트 경로에 한글이 있으므로 터미널에서 직접 실행할 때 `PYTHONUTF8=1`(PowerShell: `$env:PYTHONUTF8=1`)을 설정하세요.

## 실행 방법

### 노트북 (권장)

`notebooks/yolov3_refactored.ipynb`를 커널 **KAMP**로 열어 위에서부터 실행합니다. 흐름은 원본과 같고, 새 구조 위에서 끝까지 실행되는 것을 확인했습니다.

| 단계 | 내용 |
|---|---|
| ① | 설정, 라이브러리 (경로 설정 셀 한 곳) |
| ② | 데이터 종류/개수 확인, 호기별 재정리 |
| ③ | 라벨링 (OpenLabeling) |
| ④ | 데이터 특성 파악 |
| ⑤ | train/test 분리 (8:2), `custom.data` 생성 |
| ⑥ | 의존성 설치 |
| ⑦ | 학습 |
| ⑧ | 추론(`detect`)과 평가(`test`) |

### 명령줄

먼저 `n400`을 촬영 묶음 단위로 train/val/test = 8:1:1로 분할합니다(`data/splits/`에 저장).

```bash
python scripts/make_split.py
```

학습·추론·평가는 `third_party/yolov3`에서 실행합니다. 경로에 한글이 있어 데이터·가중치는 절대경로로 넘깁니다.

```bash
cd third_party/yolov3
python train.py --epochs 100 --weights <시작 가중치> --batch-size 8 --cfg yolov3-spp.cfg --data <KAMP>/data/splits/val.data --nosave
python detect.py --weights <가중치> --source <이미지 폴더> --cfg yolov3-spp.cfg --names <KAMP>/data/classes.names --output <KAMP>/outputs/runs/<실험명>/result
python test.py --cfg yolov3-spp.cfg --batch-size 8 --data <KAMP>/data/splits/test.data --weights <가중치>
```

- `--epochs N`은 기존 체크포인트에서 **추가로 N epoch**를 학습한다는 뜻입니다.
- `--nosave`를 써도 결과가 `third_party/yolov3/weights/last.pt`에 저장되어 덮어써집니다. 끝나면 `outputs/runs/<실험명>/`으로 옮기세요.
- GPU가 없으면 `--device cpu`를 추가합니다.

## 데이터

- 클래스: `defect` 1개, 라벨은 YOLO 포맷(`labels/*.txt`)
- `data/samples`: 실습용 15장 (train 12 / test 3)
- `data/subsets`: 15/50/100/200/300/400장 서브셋. **서로 중첩되어 고유 이미지는 400장뿐**이므로 합쳐서 학습하지 않습니다.
- `data/splits`: `n400`을 실제 호기별로 층화하고 촬영 묶음 단위로 나눈 train 320 / val 38 / test 42
- 세트별 수량, bbox 통계, 유의 사항은 [docs/dataset.md](docs/dataset.md)를 참고하세요.
- **이 저장소에는 이미지와 라벨이 포함되어 있지 않습니다**(`.gitignore`). KAMP 데이터셋을 받아 `data/subsets/n400/{images,labels}`에 배치한 뒤 `python scripts/build_manifest.py`(원본 폴더 필요) → `python scripts/make_split.py`로 분할을 만드세요. 이미지별 호기·분할 기록은 `data/manifest.csv`, `data/splits/split.csv`에 있습니다.
- 대회 공식 데이터셋(약 1.8GB)은 이 저장소에 포함되어 있지 않을 수 있습니다.

## 결과 요약

실습용 15장 데이터로 학습→추론→평가가 끝까지 동작함을 확인했습니다 (테스트 3장 / 객체 7개).

| 실행 | Precision | Recall | mAP@0.5 | F1 |
|---|---|---|---|---|
| 학습 중 검증 | 0.532 | 0.571 | 0.478 | 0.551 |
| `test.py` 단독 | 0.751 | 0.571 | 0.575 | 0.649 |

표본이 매우 작아 성능 지표로서의 의미는 제한적입니다. 자세한 내용은 [docs/experiment_results.md](docs/experiment_results.md)를 참고하세요.

## 호환성 수정

2020년 버전 yolov3 코드(`third_party/yolov3`)를 최신 PyTorch/numpy에서 실행하기 위해 `torch.load`, `np.int`, 이어서 학습하는 epoch 계산, CPU/GPU 텐서 혼용, 한글 경로 이미지 읽기 등을 수정했습니다. 원인과 해결은 [docs/troubleshooting.md](docs/troubleshooting.md)에 정리했습니다.

## 남은 과제

- `n400` 분할로 베이스라인 재학습, 비교 모델 추가 (val F1)
- 공식 대회 데이터 확보 여부 확인
- **AI 미탐지 조건 분석**: 미탐지(FN) 사례의 특성(크기, 위치, 대비 등) 분석
- 제출물 준비: 결과 보고서, 소스코드 ZIP(`requirements.txt`/`environment.yml`, 예측 결과 포함), 발표자료. 제출물에는 소속·로고 등 식별 정보를 넣지 않습니다.

## 라이선스 및 출처

- 학습 코드: [ultralytics/yolov3](https://github.com/ultralytics/yolov3) (GPL-3.0) 2020년 버전 기반
- 라벨링 도구: [OpenLabeling](https://github.com/Cartucho/OpenLabeling) (Apache-2.0)
- 데이터셋: KAMP (https://www.kamp-ai.kr)
