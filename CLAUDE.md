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
- 이 저장소의 라벨된 데이터(고유 이미지 500장)가 대회 본 데이터(약 1.8GB)와 같은지는 확인되지 않았다.

## 현재 상태 (2026-10-05)
**코드·실험 결과·문서를 모두 지우고 데이터 전처리부터 새로 시작하는 단계다.**
- `scripts/`와 `docs/`(README 제외), `runs/baseline/`은 비어 있다. 데이터 전처리는 이총이 맡아 구글 드라이브로 배포하며 링크는 미정이다.
- 지운 코드·문서·결과는 Git 이력에 있다: 코드·결과 `71813fd`, 문서 `d151bfa`. (`git show d151bfa:docs/dataset.md`, `git checkout 71813fd -- scripts/<파일>`)
- 위 이력의 수치와 문서는 **삭제 전 데이터·분할 기준**이므로 새 결과와 직접 비교하지 말 것.

## 환경
- conda 환경 **`KAMP`** (Python 3.10): torch 2.6.0+cu124, torchvision 0.21.0, numpy 2.2, opencv-python 5, jupyter (`requirements.txt`로 고정). Jupyter 커널 이름 `kamp`. 개발 PC는 RTX 4050 Laptop 6GB, Windows 11.
- **한글 경로**: 스크립트를 터미널에서 직접 실행할 때 `PYTHONUTF8=1`을 설정한다(설정 파일을 cp949로 읽어 실패). 이미지는 `cv2.imread`가 한글 경로를 못 읽으므로 `cv2.imdecode(np.fromfile(...))`로 읽는다(`src/yolov3/utils/datasets.py`의 `imread`/`imwrite` 헬퍼).

## 디렉터리 구조와 영역
**공유 영역(함부로 바꾸지 말 것, PR + 팀 확인)**: `data/`, `scripts/`, `src/yolov3/`, `runs/baseline/`
**개인 영역(본인만 수정, 다른 사람 폴더는 건드리지 말 것)**: `src/<이름>/`, `notebooks/<이름>/`, `runs/<이름>/` (`<이름>` = `minyeop`, `yeonchang`, `chong`)

- `data/` — 데이터와 목록. 이미지·라벨은 Git에 없고 이총이 드라이브로 배포한다. `classes.names`(`defect` 1개)만 들어 있다.
- `scripts/` — 전처리·분할·학습 실행·평가 파이프라인 (현재 비어 있음, 새로 작성).
- `src/yolov3/` — 호환성 수정을 거친 YOLOv3 코드(`train.py`, `test.py`, `detect.py`, `models.py`, `utils/`, `yolov3-spp.cfg`, `yolov3-tiny.cfg`). 데이터·가중치는 포함하지 않는다. 출처·수정 내역은 `src/yolov3/README.md`.
- `notebooks/shared/` — 원본 실습 노트북 2개(경로 하드코딩, 장비 표시가 있는 이미지로 학습하는 흐름이라 참고용).
- `weights/` — Git 제외. COCO 사전학습 가중치는 `weights/README.md`의 방법으로 받는다.
- 원본 `4. X-ray 검사장비 AI 데이터셋/dataset/` — 수정하지 않는 읽기 전용 출처. `test1/yolov3/X선이물검출기(06.23_09.22)/`에 원본 bmp 2,809장이 있다. 탐색 시 제외할 것.
- 폴더마다 README가 있어 역할과 규칙을 설명한다.

## 데이터 주의 (이전 시도에서 확인한 사실, 근거는 `git show d151bfa:docs/dataset.md`)
- **이미지에 장비가 그린 색 박스(표시)가 찍혀 있다**(확인한 500장 모두 1~3개, 정답 중심의 99.8%가 표시 안). 지우지 않고 학습하면 모델이 표시를 보고 맞힌다. 표시를 지운 입력에서 그런 모델은 TP 0이었다. Navier-Stokes 보간 + 주변과 같은 세기의 노이즈(x0.75)로 지웠을 때 윤곽 단서가 가장 작았다(질감 AUC 0.57, 노이즈 없이는 0.73~0.75).
- **라벨된 이미지는 500장**: 라벨 txt 500개는 `라벨링 6종 세트/labels/`, 이미지 400장은 `images 400/`, 나머지 100장은 원본 bmp 폴더에 같은 이름으로 있다.
- **파일명 접두 001/002는 호기가 아니다**(002 = 1·2호기, 001 = 3호기). 호기는 원본 폴더 위치로만 안다. 해상도는 호기별로 다르다(352x332 / 316x332 / 576x444).
- 모든 이미지가 NG 판정이라 **정상 이미지는 없다.** 원본 bmp 2,809장 중 277장은 같은 호기 내 중복(고유 2,532장).
- 라벨은 이미지 경로의 `/images/`를 `/labels/`로 바꾼 위치의 같은 이름 `.txt`(YOLO 정규화 포맷)를 코드가 찾으므로 **`images/`와 `labels/`는 형제 폴더**여야 한다.
- 분할은 호기별 층화 + 연속 촬영(60초 이내) 묶음 단위로 해야 누수가 없다.
- 대용량 `.pt`와 `*.bmp`가 많으므로 전체 디렉터리를 무분별하게 Glob/Grep/Read 하지 말 것.

## 구버전 코드 수정 이력 (`src/yolov3`, 되돌리지 말 것)
최신 PyTorch/numpy/Windows 호환을 위해 수정했다.
- 모든 `torch.load(...)`에 `weights_only=False`
- `np.int` → `int`
- `train.py`: 이어서 학습 시 `epochs += start_epoch`(기존 `ckpt['epoch']`는 반복 0회 버그), 환경변수 `YOLO_SAVE_DIR`로 산출물 위치 지정, DataLoader `persistent_workers`
- `utils/utils.py`: `build_targets`의 `torch.arange(..., device=targets.device)`, `output_to_target`의 `float()` 변환
- `utils/datasets.py`: 한글 경로용 `imread`/`imwrite` 헬퍼, `detect.py`에서도 사용

## 학습·평가 시 주의
- `src/yolov3/train.py`의 `--data`/`--cfg` 기본값은 COCO용이라 항상 명시한다. `--epochs N`은 기존 체크포인트에서 **추가 N epoch**를 뜻한다.
- `YOLO_SAVE_DIR` 없이 학습하면 `src/yolov3/weights/last.pt`가 덮어써진다(`--nosave`여도 마지막 epoch는 저장). 실험마다 출력 폴더를 분리한다.
- **평가는 신뢰도 임계값을 val에서 정해 고정하고 test는 한 번만 본다.** `test.py`의 P/R/F1은 신뢰도 0.1 한 지점 값이라 쓰지 않는다. 이전 모델은 신뢰도가 낮아 0.5 이상에서 아무것도 못 찾았다.
- 시작 가중치는 COCO 사전학습본을 쓴다. 이전 학습본으로 시작하면 평가에 누수가 생길 수 있다.
- 프로젝트가 OneDrive 폴더 안이면 상태 파일 교체가 거부될 수 있고(WinError 5), 긴 학습 중 PC 절전이 학습을 멈춘 적이 있다.
