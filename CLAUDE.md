# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

KAMP "X-ray 검사장비 AI 데이터셋" 프로젝트. X선 이물 검출기 영상에서 결함(`defect`, 단일 클래스)을 검출한다. 베이스라인 코드는 `src/yolov3`(ultralytics YOLOv3 2020년 버전을 수정한 사본)에 있다. 빌드·린트·자동 테스트는 없다. **3명(허민엽 `minyeop`, 정연창 `yeonchang`, 이총 `chong`)이 협업**하며 규칙은 `CONTRIBUTING.md`에 있다.

### 대회 맥락 (KAMP 공지 NOTICE_SEQ=86 기준)
"2026년 제6회 K-인공지능 제조데이터 분석 경진대회" 과제 중 하나가 이 프로젝트의 주제다. 공지에 명시된 내용만 정리한다.
- 과제: **영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석** (X-ray 검사장비 AI 데이터셋, 약 1.8GB)
- 일정: 과제해결 2026-09-21 ~ **2026-10-08 23:59**, 서면평가 결과 10-26, 발표평가 10-30
- 제출물: 결과 보고서(PDF, 제공 양식), 소스코드(ZIP: `requirements.txt` 또는 `environment.yml`, 학습용 데이터, README, 테스트데이터 예측결과 포함), 발표자료(PDF+PPT), 설문(네이버 폼 완료 화면을 보고서에 첨부)
- **모든 제출물에 소속(기업·학교명)·로고 등 식별 정보 기재 금지**, 성명/팀명만 허용. 코드·문서·커밋에 소속을 넣지 말 것.
- 서면평가 배점(100점): 데이터 이해·진단 15 / AI 모델 개발 **40**(베이스라인 포함 2개 이상 비교, F1 등) / 영향요인·오류분석 15(FN·FP 집중 조건) / 현장 활용방안 10 / 창의성 10 / 코드·재현성 10(전처리→학습→추론→결과 자동 실행). 이 표가 X-ray 전용인지는 확인되지 않았다.
- 이 저장소의 데이터(고유 이미지 2,532장)가 대회 본 데이터(약 1.8GB)와 같은지는 확인되지 않았다.

## 현재 상태 (2026-10-05)
**데이터는 확정 v2이고, 코드·실험 결과·문서는 새로 만드는 단계다.**
- `data/`에 확정 데이터 v2가 있다(아래 "데이터").
- `scripts/`, `runs/baseline/`은 비어 있다. `docs/`에는 `metrics.md`(지표), `experiments.md`(모델 공통 결과 표), `README.md`(결과 기록 가이드)가 있다.
- 첫 모델로 **Faster R-CNN**(`src/minyeop/faster_rcnn/`)을 학습·평가했다. 아래 "Faster R-CNN" 절.
- 지운 코드·문서·결과는 Git 이력에 있다: 코드·결과 `71813fd`, 문서 `d151bfa`. (`git show d151bfa:docs/dataset.md`, `git checkout 71813fd -- scripts/<파일>`)
- 위 이력의 수치와 문서는 **이전 데이터(라벨 500장)·분할 기준**이므로 새 결과와 직접 비교하지 말 것.

## 환경
- conda 환경 **`KAMP`** (Python 3.10): torch 2.6.0+cu124, torchvision 0.21.0, numpy 2.2, opencv-python 5, jupyter (`requirements.txt`로 고정). Jupyter 커널 이름 `kamp`. 개발 PC는 RTX 4050 Laptop 6GB, Windows 11.
- **한글 경로**: 스크립트를 터미널에서 직접 실행할 때 `PYTHONUTF8=1`을 설정한다(설정 파일을 cp949로 읽어 실패). 이미지는 `cv2.imread`가 한글 경로를 못 읽으므로 `cv2.imdecode(np.fromfile(...))`로 읽는다(`src/yolov3/utils/datasets.py`의 `imread`/`imwrite` 헬퍼).

## 디렉터리 구조와 영역
**공유 영역(함부로 바꾸지 말 것, PR + 팀 확인)**: `data/`, `scripts/`, `src/yolov3/`, `runs/baseline/`
**개인 영역(본인만 수정, 다른 사람 폴더는 건드리지 말 것)**: `src/<이름>/`, `notebooks/<이름>/`, `runs/<이름>/` (`<이름>` = `minyeop`, `yeonchang`, `chong`)

- `data/` — 확정 데이터 v2(이미지, 라벨, 분할, 매니페스트, 조건 파일). 저장소를 **비공개로 두는 전제**로 Git에 둔다(공개로 바꾸면 KAMP 데이터가 공개됨, 재배포 조건 미확인).
- `scripts/` — 전처리·학습 실행·평가 파이프라인 (현재 비어 있음, 새로 작성).
- `src/yolov3/` — 호환성 수정을 거친 YOLOv3 코드(`train.py`, `test.py`, `detect.py`, `models.py`, `utils/`, `yolov3-spp.cfg`, `yolov3-tiny.cfg`). 데이터·가중치는 포함하지 않는다. 출처·수정 내역은 `src/yolov3/README.md`.
- `src/minyeop/faster_rcnn/` — 허민엽의 Faster R-CNN(torchvision, ResNet-50 FPN) 학습·예측·보고·시각화 코드. 결과는 `runs/minyeop/01_frcnn_r50fpn_min640_v1/`.
- `notebooks/shared/` — 원본 실습 노트북 2개(경로 하드코딩, 장비 표시가 있는 이미지로 학습하는 흐름이라 참고용).
- `weights/` — Git 제외. COCO 사전학습 가중치는 `weights/README.md`의 방법으로 받는다.
- 원본 `4. X-ray 검사장비 AI 데이터셋/dataset/` — 수정하지 않는 읽기 전용 출처(있는 PC에서만). 원본 bmp 2,809장은 `test1/yolov3/X선이물검출기(06.23_09.22)/`에 있다. 탐색 시 제외할 것.
- 폴더마다 README가 있어 역할과 규칙을 설명한다(`data/`는 `README.md`).

## 데이터 (확정 v2, 2026-10-05 직접 검증, 설명 원문은 `data/README.md`)
- 구조: `data/{train,val,test}/{images/*.png, labels/*.txt}`, `manifest.csv`, `conditions.csv`, `conditions_thresholds.json`, `split_info.json`, `manifest.sha256`, `PASS`, `README.md`.
- 이미지 2,532장 전부 라벨 있음. **train 1,767 / val 369 / test 396**, 박스 4,494개(3,225 / 606 / 663). 이미지는 **회색조 PNG**이고 장비 색 박스(표시)가 지워져 있다(표본 300장에서 색 픽셀 0).
- 라벨: YOLO txt `0 cx cy w h`(0~1 비율), 클래스 0 = defect. 코드가 이미지 경로의 `/images/`를 `/labels/`로 바꿔 라벨을 찾으므로 **`images/`와 `labels/`는 형제 폴더**여야 한다.
- **버전**: `manifest.csv`의 sha256 = `1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623`(`PASS`에 기록). 다른 값이면 다른 버전이다.
- **분할은 다시 만들지 말 것**: 촬영 묶음(burst, 60초 이내 연속) 단위 무작위, test 0.15 / val 0.15 / 나머지 train, seed 42. 바꾸면 모든 결과가 비교 불가가 된다.
- 호기별: 1호기 943장(train 672 / val 134 / test 137), 2호기 805장(573 / 102 / 130), 3호기 784장(522 / 133 / 129). 해상도: 1호기 316x332 431·352x332 392·412x332 120장, 2호기 316x332 805장, 3호기 576x444 784장 — **해상도는 호기별 고정이 아니다**. 파일명 접두 001/002는 호기가 아니다(호기는 `manifest.csv`의 `machine`).
- **라벨 두 종류**: 공식 라벨 500장(KAMP 제공, 박스 1,147개, 사람이 그려 크기가 실제 결함을 반영, 한 변 5~21px)과 팀 라벨 2,032장(heo/jung/lee, 박스 3,347개, **클릭 중심에 같은 크기 네모 자동 생성**: 1·2호기 10px, 3호기 약 13px 고정). 팀 라벨에는 결함 크기 정보가 없고, 작은 박스(약 10px)의 IoU는 중심 오차에 민감하다. 공식 라벨은 val 59장·test 85장.
- **빈 라벨 105장**(train 59 / val 19 / test 27): 제품이 반만 찍혀 테스트피스가 없는 사진. 이물질 없음 사진으로 쓸 수 있으나 일반 정상 제품 이미지는 아니다. 그 외 모든 이미지는 장비가 NG로 판정한 것이다.
- `conditions.csv`(박스 4,494행): 크기, 대비, 배경 밝기, 가장자리 거리, 호기, 해상도, 월, 출처, 작성자와 구간. `size_bin`은 팀 라벨이 고정 크기라 호기를 반영하고, `edge_dist_bin`은 near가 8개뿐이라 쓸 수 없다.
- 계보: KAMP 원본 BMP → 중복 제거 → 장비 색상 박스 제거(주변 회색 메움) → 회색조 PNG. 가공 코드는 이 저장소에 없다.
- **이전 시도에서 확인한 사실**(근거 `git show d151bfa:docs/dataset.md`): 표시가 있는 이미지로 학습한 모델은 표시를 지운 입력에서 TP 0이었다. 주변 픽셀로 메우기만 하면 윤곽이 남고(질감 AUC 0.73~0.75) 주변과 같은 노이즈를 더하면 줄었다(0.57). **v2의 윤곽 흔적은 검증하지 않았다.**
- 대용량 `.pt`와 데이터 이미지가 많으므로 전체 디렉터리를 무분별하게 Glob/Grep/Read 하지 말 것.

## 라벨링 툴 (팀이 직접 제작했던 방식, 참고)
- 팀 라벨은 클릭 중심 + 해상도 그룹별 고정 크기 네모로 만들었다. 이 방식은 결함 크기 정보를 없애므로 크기별 미탐지 분석에는 공식 라벨만 쓴다.
- 평가를 IoU 대신 "예측 중심이 정답 중심에서 몇 px 이내인가"로 보는 방법도 고려할 만하다(시험해 보지 않음).

## Faster R-CNN (`src/minyeop/faster_rcnn/`, 허민엽)
- 파일: `dataset.py`(YOLO txt → xyxy, 회색조 3채널, 빈 라벨 포함), `model.py`(COCO 사전학습, 2클래스 헤드, `--min-size`/`--anchor-sizes`), `metrics.py`(AP·P/R/F1), `train.py`, `predict.py`, `report.py`(보고 표 계산), `watch.py`(진행 모니터), `visualize.py`(그림).
- 실행(`PYTHONUTF8=1`): `train.py --name <실험> --epochs 20 --batch-size 4`, `watch.py --name <실험> --follow`, `predict.py --name <실험> --split val|test --ckpt best`, `report.py --name <실험>`, `visualize.py --name <실험> --split val|test`. 결과는 `runs/minyeop/<실험>/`. `train.py`는 test를 쓰지 않고, 실험 폴더가 이미 있으면 중단한다. `predict.py --split test`는 같은 체크포인트의 val 결과로 임계값을 정하므로 먼저 val을 같은 `--ckpt`로 돌릴 것(이전 체크포인트의 `preds_val.json`이 남아 있으면 임계값이 어긋남).
- 결과(`01_frcnn_r50fpn_min640_v1`, 입력 640, epoch 7): test F1 0.989(공식 라벨만 0.964, 팀 라벨 1.000), val F1 0.992, 임계값 0.95, 추론 약 31ms/장, 학습 1시간 07분.
- **점수 해석 주의**: 팀 라벨이 고정 크기라 IoU 0.5 기준이 쉽게 포화된다. 오류는 대부분 크기 차이로 인한 IoU 미달이고(같은 점을 중심 1.2px 이내로 찾음) 진짜 미탐지가 아니다. 결함이 막대 끝에 있는 규칙성, 표시 흔적을 모델이 지름길로 쓰는지는 검증하지 않았다.
- 가중치(`weights/`), 예측 결과(`preds_*.json`, 정답 좌표 포함), `status.json`, `progress.log`는 Git에 올리지 않는다. 그림(`figures/`)에는 데이터 이미지가 그려져 있어 비공개 저장소에서만 올린다.

## 그 밖의 도구와 분석 (허민엽, `src/minyeop/`)
- `yolov3_tiny/`: 공유 `src/yolov3`를 수정하지 않고 실험 폴더 안에서 학습하는 `run.py`, `predict.py`(공통 JSON 저장), `watch.py`. 결과 `runs/minyeop/02_yolov3tiny_img640_v1`(test F1 0.989, 임계값 0.06, 학습 59분). `train.log`에 개인 PC 경로가 있어 Git에 올리지 않는다.
- `faster_rcnn_mobilenet/`: `faster_rcnn`의 학습·예측 코드를 재사용하고 모델만 `fasterrcnn_mobilenet_v3_large_fpn`으로 교체(COCO 사전학습, `--anchor-sizes`는 5개 크기를 3레벨에 적용). 이 FPN은 stride 16 이상만 쓴다. 결과 `runs/minyeop/04_frcnn_mobv3_min640_anc16_v1`: test F1 0.986(공식 라벨 0.959), 임계값 0.93, 학습 20분. 찾는 능력은 ResNet-50과 같은 수준이고 박스 정밀도만 낮다(mAP50-95 0.559 대 0.611). 같은 조건에서 추론 약 4배 빠름.
- `faster_rcnn/report.py --center-r 5 --tag _v2`: 평가 v2(중심 거리 매칭, 임계값도 이 기준으로 val에서 재선정). R≥3px면 두 모델 모두 test 만점이라 구분은 R=1~2px에서만 된다. IoU 0.5 기준 F1은 사실상 중심 2px 이내 여부를 재고 있었다.
- `dot_removal/remove_dot.py`: test 정답의 어두운 점을 보간(Navier-Stokes 또는 평균)으로 지운 뒤 두 모델에 다시 넣는 지름길 검증. 점을 충분히 지우면 검출이 거의 0이 된다(`runs/minyeop/03_dot_removal_v1/`, 해석은 `docs/README.md`).
- `fake_normal/false_alarm.py`: 점을 지운 test 이미지(가짜 정상)에서 세 모델의 이미지 단위 오경보를 계산한다(`runs/minyeop/05_fake_normal_v1/`). 평균 보간은 0.8% 이하, 노이즈를 더한 보간은 YOLO·MobileNet 22~64%라 **오경보는 범위로 해석**한다(진짜 정상 제품이 아님).
- `synth_insert/insert_dots.py`: val의 실제 점을 test 바탕에 s·f를 바꿔 합성해 검출률을 본다(`runs/minyeop/06_synth_insert_v1/`). MobileNet은 무작위 자리에서 30%(R50·YOLO 약 90%)라 위치 의존이 의심되지만, 합성 점의 사각형 자국이 교란 요인이다.
- `group_stats/`: 호기·해상도별 통계(`runs/minyeop/07_group_stats_v1/`). 해상도별 공식 라벨 비율이 달라(10~76%) 순수한 호기·해상도 효과를 분리하지 못한다.
- 이 세 분석과 속도 측정의 해석·한계·test 사용 이력은 `docs/analysis.md`. 강건성 시험은 test 이미지로 먼저 했으므로 모델 선정 근거로 쓰려면 val로 반복해야 한다.

## 구버전 코드 수정 이력 (`src/yolov3`, 되돌리지 말 것)
최신 PyTorch/numpy/Windows 호환을 위해 수정했다.
- 모든 `torch.load(...)`에 `weights_only=False`
- `np.int` → `int`
- `train.py`: 이어서 학습 시 `epochs += start_epoch`(기존 `ckpt['epoch']`는 반복 0회 버그), 환경변수 `YOLO_SAVE_DIR`로 산출물 위치 지정, DataLoader `persistent_workers`
- `utils/utils.py`: `build_targets`의 `torch.arange(..., device=targets.device)`, `output_to_target`의 `float()` 변환
- `utils/datasets.py`: 한글 경로용 `imread`/`imwrite` 헬퍼, `detect.py`에서도 사용
- `utils/datasets.py`(2026-10-06): 이미지가 1,000장을 넘으면 만드는 라벨 캐시 저장(`np.save`)이 numpy 2에서 실패해서(길이가 다른 배열 목록), object 배열로 직접 만들어 저장하도록 수정. 캐시는 `data/<split>/labels.npy`로 생기며 `.gitignore`의 `*.npy`로 제외. 학습 동작에는 영향 없음

## 학습·평가 시 주의
- `src/yolov3/train.py`의 `--data`/`--cfg` 기본값은 COCO용이라 항상 명시한다. `--epochs N`은 기존 체크포인트에서 **추가 N epoch**를 뜻한다.
- `YOLO_SAVE_DIR` 없이 학습하면 `src/yolov3/weights/last.pt`가 덮어써진다(`--nosave`여도 마지막 epoch는 저장). 실험마다 출력 폴더를 분리한다.
- **평가는 신뢰도 임계값을 val에서 정해 고정하고 test는 한 번만 본다.** `test.py`의 P/R/F1은 신뢰도 0.1 한 지점 값이라 쓰지 않는다. 이전 모델은 신뢰도가 낮아 0.5 이상에서 아무것도 못 찾았다. 지표 목록과 주의는 `docs/metrics.md`.
- 모델 공통 결과는 `docs/experiments.md`의 표에 **본인 행만** 기록하고, 규칙·검출 결과 JSON 형식은 `docs/README.md`를 따른다.
- 시작 가중치는 COCO 사전학습본을 쓴다. 이전 학습본으로 시작하면 평가에 누수가 생길 수 있다.
- 프로젝트가 OneDrive 폴더 안이면 상태 파일 교체가 거부될 수 있고(WinError 5), 긴 학습 중 PC 절전이 학습을 멈춘 적이 있다.
