# 전처리 파이프라인: KAMP 원본 → 확정 데이터 v2 (기록)

> 제출 저장소 안내: 이 폴더는 원래 `scripts/preprocess/` 에 있던 코드를 그대로 옮긴 것입니다. 본문의 `scripts/preprocess/...`, `src/chong/harness/...` 경로는 각각 `preprocessing/...`, `experiments/comparison_models/yolo26/...` 로 읽으면 됩니다. 결과가 `data/` 와 같은지는 `python scripts/verify_preprocess.py --raw <원본 폴더>` 로 확인합니다 (2,532장 픽셀 단위 일치).

`data/` 에 있는 확정 데이터 v2 (2,532장, train/val/test) 를 KAMP 원본에서 어떻게 만들었는지 설명하고, 그때 쓴 코드를 그대로 둔 폴더입니다.
보고서의 "데이터 이해·진단" 과 "코드·재현성" 항목의 근거입니다. 코드는 2026-10-02 ~ 10-05 에 실제로 돌린 것이고, 지금의 `data/` 는 확정본이라 **다시 만들지 않습니다** (다시 만들면 manifest 해시가 바뀌어 모든 결과가 비교 불가).

## 한눈에 보기

| 단계 | 무엇을 | 왜 | 코드 | 결과 |
|---|---|---|---|---|
| 0 | 원본 받기 | | (KAMP 배포본) | BMP 2,809장 + 라벨 txt 500개 |
| 1 | 사진 목록 만들기, 중복 제거, 묶음(burst) 표시 | 같은 사진이 277장 겹쳐 있음. 연속 촬영본이 거의 같아 분할 단위가 필요 | `build_manifest.py` | `manifest.csv` 2,532장 |
| 2 | 장비 색상 박스 지우기, 회색조 PNG 로 | 라벨 중심의 99.8% 가 색상 박스 안 → 안 지우면 "색 네모 찾기" 를 배움 | `preprocess.py fill` | 사진 2,532장 (v1_fill) |
| 3 | 팀 라벨링 (라벨 없는 2,032장) | 학습 데이터가 500장뿐. 나머지도 같은 시편 사진이라 라벨 가능 | `labeling/` | 팀 라벨 2,032개 (빈 라벨 105) |
| 4 | 조건 축 계산 | 오류분석(크기·대비·배경·가장자리·호기별) | `src/chong/harness/common/compute_conditions.py` | `conditions.csv` |
| 5 | 분할 (train/val/test) | 묶음 단위로 나눠야 같은 제품이 양쪽에 안 들어감 | `make_splits_v2.py` | `manifest.csv` 의 `split` 열, 폴더 고정 |
| 6 | 검사·잠금 | 세 사람이 같은 데이터인지 확인 | `check_data_version.py` (→ 지금은 `common/check_data.py`) | `manifest.sha256`, `PASS` |

## 단계별 설명

### 1. 사진 목록과 중복 제거 (`build_manifest.py`)
- 원본 폴더 `N호기/날짜/…` 를 훑습니다. **장비는 폴더 이름(N호기)으로** 정합니다. 파일명 앞의 `001_`/`002_` 는 장비가 아닙니다.
- 촬영 시각은 파일명 `NNN_YYYYMMDD_HHMMSS(N)` 에서 읽습니다. 괄호 숫자 `(N)` 은 상자 수가 아닙니다(일치율 0.108). 뜻은 모르고 보관만 합니다.
- 파일 내용의 sha256 이 같은 사진은 하나만 남깁니다 (277장 제거 → 2,532장). 확장자가 `.jpg` 인 라벨 사진도 실제로는 BMP 라서 해시로 원본과 짝지었습니다.
- **묶음(burst)**: 같은 장비에서 60초 이내로 이어진 사진을 한 묶음으로 봅니다 (`burst_id`). 6장 묶음이 336개로 가장 흔하고, 같은 묶음의 이웃 사진 상관이 중앙값 0.90 으로 거의 같은 사진입니다. 그래서 분할은 묶음 단위입니다.

### 2. 색상 박스 지우기 (`preprocess.py fill`)
- 원본은 8비트 팔레트 사진이고 회색은 R=G=B 입니다. `max(RGB) − min(RGB) > 0` 인 픽셀이 장비가 그린 색상 박스입니다 (사진의 0.05~0.5%).
- 그 픽셀을 **주변 회색 픽셀의 5×5 평균으로 바깥쪽부터 채웁니다** (`fill_neighbors`, 채워질 때까지 반복). 결과는 회색조 PNG, 원본과 같은 크기.
- 왜 지우나: 공식 라벨 중심의 99.8% 가 색상 박스 안에 있어, 안 지우면 모델이 이물질이 아니라 색 네모를 찾습니다. 공지도 "표시는 특징으로 쓰지 말 것" 이라고 합니다. **라벨 없는 사진에도 색상 박스가 있으며**(장비가 불량 판정한 사진만 저장되기 때문), 그 박스를 가짜 라벨로 쓰면 안 됩니다. 정답은 txt 뿐입니다.
- 후보 비교: `raw`(안 지움), `fill`(주변 평균), `inpaint`(OpenCV Telea), `mask`(단색 덮기) 네 가지를 만들 수 있고 `fill` 을 확정했습니다. 메운 자리의 흔적이 새 단서가 될 수 있다는 점은 알려진 한계입니다 (이전 시도에서 흔적 구분 AUC 0.73~0.75). v2 의 흔적 반응은 `src/chong` 의 합성 정상 사진 검사에서 "임의 자리를 같은 방식으로 메워도 모델이 반응하지 않음(0%)" 으로 확인했습니다.

### 3. 팀 라벨링 (`labeling/`)
- 라벨 없는 2,032장도 같은 시편 사진이라 이물질이 있습니다. 세 사람이 나눠 찍었습니다.
- `make_assignments.py`: 묶음 단위로 세 사람에게 나눕니다 (장비×월 비율 유지, 같은 묶음은 한 사람이). 겹침 54장은 세 명이 모두 찍어 일치율을 재는 용도입니다.
- 라벨링 서버: Label Studio 를 한 PC 에서 띄우고(`start_server.bat`, `setup_projects.py`) 터널로 공유했습니다. 팀원은 설치 없이 브라우저에서 **이물질 한가운데를 클릭**만 합니다. `fixed_box_backend.py` 가 클릭 지점에 **고정 크기 네모**를 자동으로 붙입니다: 공식 라벨 한 변 중앙값 10px 를 사진 높이에 비례시켜 1·2호기 10px, 3호기 13px. 찍는 법은 `LABELING.md`.
- `export_labels.py` → YOLO txt (`0 cx cy w h`, 0~1 비율) 로 내보내고 `import_labels.py` 로 저장소에 넣었습니다.
- **그래서 팀 라벨의 상자 크기에는 이물질 크기 정보가 없습니다.** 크기별 분석은 공식 라벨 500장으로만 합니다. 또한 작은 상자(10px)의 IoU 는 중심이 2px 만 어긋나도 크게 떨어지므로, 공식 라벨에서 IoU 기준 재현율이 낮게 나오면 "못 찾음" 이 아니라 "크기 불일치" 일 수 있습니다 (중심 적중 지표를 같이 보세요).
- 빈 라벨 105장: 제품이 반만 찍혀 시편이 없는 사진 (7/27 1호기 104장 등). "이물질 없음" 으로 쓸 수 있지만 보통의 정상 제품 사진은 아닙니다.

### 4. 조건 축 (`compute_conditions.py`, `src/chong/harness/common`)
상자마다 크기(`box_side_px`), 국소 대비(`local_contrast`), 배경 밝기(`ring_mean`), 가장자리 거리, 호기, 해상도, 월, 라벨 출처, 작성자와 그 구간(`size_bin` 등)을 계산해 `data/conditions.csv` 에 둡니다. 경계값은 `conditions_thresholds.json`. 주의: `size_bin` 은 팀 라벨이 고정 크기라 사실상 호기를 반영하고, 가장자리 `near` 는 8개뿐입니다.

### 5. 분할 (`make_splits_v2.py`)
- 2,532장 전부(공식 + 팀 라벨)를 **묶음 단위 무작위**로 나눕니다: test 15%(396장), val 15%(369장), train 나머지(1,767장), seed 42, `GroupShuffleSplit`. 같은 묶음은 한쪽에만 들어갑니다.
- val 은 평소 실험의 채점과 임계값 결정에, test 는 최종 후보 한 번에만 씁니다.
- 결과는 `manifest.csv` 의 `split` 열이고, 사진·라벨도 `data/train|val|test/` 폴더로 고정했습니다 (ultralytics 등이 `images/`↔`labels/` 형제 폴더를 기대).

### 6. 검사와 잠금
`manifest.csv` 의 sha256 (`1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623`) 이 같으면 같은 데이터입니다. `check_data_version.py` 가 사진 수·크기·라벨 짝·색 픽셀 0 을 검사해 `PASS` 를 만듭니다 (지금 구조에 맞는 판은 `src/chong/harness/common/check_data.py`).

## 모델링할 때 알아 둘 데이터 사실
1. **같은 시편을 반복 촬영한 데이터입니다.** 제품은 프레임 안에서 움직이지만 이물질 3개는 제품 기준 항상 같은 자리(어두운 띠 왼쪽 끝)에 세로로 있습니다. 모델이 "그 자리" 를 외워도 점수가 나오므로, 높은 점수를 일반 성능으로 읽으면 안 됩니다. (검증: `runs/chong/01_yolo26n_img1024/synth_normal.md`)
2. **장비가 불량이라고 판정한 사진만 저장됩니다.** 정상 사진이 0장인 이유이고, 장비가 놓친 이물질은 데이터에 없습니다. 라벨 없는 사진의 색상 박스도 같은 이유로 생긴 것이라 라벨로 쓰면 안 됩니다.
3. 해상도는 호기별 고정이 아닙니다 (1호기 316/352/412×332 혼재, 2호기 316×332, 3호기 576×444). 2호기가 어둡고 대비가 큽니다. 라벨의 59% 가 6월입니다.
4. 이물질 상자는 한 변 5~21px(중앙값 10px) 로 전부 작고, 가장자리 사례가 없습니다.

## 다시 돌리려면 (기록용)
원본이 있는 PC 에서만 됩니다. `data/raw_path.txt` 에 원본 폴더 경로를 적고, 이 폴더의 `kx.py`(당시 공용 도우미) 가 같은 폴더에 있어야 합니다.
```
PYTHONUTF8=1 python scripts/preprocess/build_manifest.py
PYTHONUTF8=1 python scripts/preprocess/preprocess.py v1_fill
PYTHONUTF8=1 python scripts/preprocess/check_data_version.py v1_fill
(팀 라벨링 → labeling/ 의 순서대로)
PYTHONUTF8=1 python scripts/preprocess/make_splits_v2.py --version v2 --seed 42
```
당시 폴더 구조(`data/versions/<버전>/`, `data/splits/`)는 지금의 `data/train|val|test` 로 바뀌었습니다. 그 전환 과정은 `LINEAGE.md` 와 하네스 저장소 이력에 있습니다.
