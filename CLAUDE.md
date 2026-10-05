# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

KAMP "X-ray 검사장비 AI 데이터셋" 프로젝트. X선 이물 검출기 영상(`.bmp`/`.jpg`)에서 결함(`defect`, 단일 클래스)을 검출하도록 **YOLOv3-SPP**(ultralytics 2020년 버전 PyTorch 구현)를 학습/추론/평가한다. 빌드·린트·자동 테스트는 없다. **3명(허민엽 `minyeop`, 정연창 `yeonchang`, 이총 `chong`)이 협업**하며 규칙은 `CONTRIBUTING.md`에 있다. ML 프로젝트 표준 구조(`docs/structure.md`)로 정리되어 있으며, 원본 실습 폴더 `4. X-ray 검사장비 AI 데이터셋/`는 **원본 데이터 출처**다(더 이상 수정하지 말 것. 이전 작업으로 일부가 이미 변경됨: `docs/original_folder_inventory.md`). 작업은 아래 표준 구조 폴더에서 한다.

### 대회 맥락 (KAMP 공지 NOTICE_SEQ=86 기준)

"2026년 제6회 K-인공지능 제조데이터 분석 경진대회" 과제 중 하나가 이 프로젝트의 주제다. 공지에 명시된 내용만 정리한다.

- 과제: **영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석** (X-ray 검사장비 AI 데이터셋, 약 1.8GB)
- 일정: 과제해결 2026-09-21 ~ **2026-10-08 23:59**, 서면평가 결과 10-26, 발표평가 10-30
- 제출물: 결과 보고서(PDF, 제공 양식), 소스코드(ZIP: `requirements.txt` 또는 `environment.yml`, 학습용 데이터, README, 테스트데이터 예측결과 포함), 발표자료(PDF+PPT), 설문(네이버 폼 완료 화면을 보고서에 첨부)
- **모든 제출물에 소속(기업·학교명)·로고 등 식별 정보 기재 금지**, 성명/팀명만 허용. 소스·README·문서 작성 시 소속을 넣지 말 것.
- 개발환경: 자체(Python/Anaconda-Jupyter), KAMP-NOTE, KAMP AI-PaaS 중 선택.
- 서면평가 배점(100점): 데이터 이해·진단 15 / AI 모델 개발 **40**(베이스라인 포함 2개 이상 비교, F1 등) / 영향요인·오류분석 15(FN·FP 집중 조건) / 현장 활용방안 10 / 창의성 10 / 코드·재현성 10(전처리→학습→추론→결과 자동 실행). 상세와 현재 상태는 `docs/evaluation.md`. 이 표가 X-ray 전용인지는 확인되지 않았다.
- 이 저장소의 라벨된 데이터(고유 이미지 500장 등)는 실습용 소규모 세트이며, 대회 본 데이터(약 1.8GB)와 동일한지는 확인되지 않았다. "AI 미탐지 조건 분석"은 아직 구현되어 있지 않다.

## 환경

- conda 환경 **`KAMP`** (Python 3.10, `C:\Anaconda\envs\KAMP`): torch 2.6.0+cu124, torchvision 0.21.0, numpy 2.2, opencv-python 5, jupyter. Jupyter 커널 이름 `kamp`. GPU는 RTX 4050 Laptop(6GB).
- 한글 경로 때문에 터미널에서 스크립트를 직접 실행할 때 `PYTHONUTF8=1`을 설정해야 한다(설정 파일을 cp949로 읽어 실패함). 노트북은 이를 자동 설정한다.

## 디렉터리 구조 (상세: `docs/structure.md`, 규칙: `CONTRIBUTING.md`)

**공유 영역(함부로 바꾸지 말 것, PR + 팀 확인)**
- `data/` — `manifest.csv`(이미지별 실제 호기·원본 폴더·크기·bbox 수), `classes.names`, `splits/`(n500을 호기별 층화 + 촬영 묶음 단위로 나눈 train/val/test = 400/51/49, bbox 914/117/116, `scripts/make_split.py` 산출물), `splits_nomark/`(표시 제거본용 목록, 분할 배정은 동일), `archive/`(이전 n400·접두 기준 분할, 참고용). 이미지·라벨은 Git에 없다: `subsets/n500`(원본 이미지), `nomark/`(표시를 지운 이미지, 학습에 쓰는 것), `samples/`, `preview_nomark/`.
- `scripts/` — 데이터 준비(`make_split`, `build_manifest`, `check_data`, `remove_marks`), 실험(`run_experiment`, `watch_experiment`, `reeval_experiment`), 평가·검증(`evaluate`, `check_ring_cue`).
- `src/yolov3/` — 호환성 수정을 거친 YOLOv3 학습/추론/평가 코드(베이스라인). 데이터·가중치는 포함하지 않는다. 출처·라이선스는 `src/yolov3/README.md`.
- `runs/baseline/` — 팀 기준 베이스라인 결과(`03_yolov3tiny_nomark`, `04_yolov3spp_nomark`)와 `_invalid/`(표시가 있던 이전 실험, 중단·불완전 실행).

**개인 영역(본인만 수정, 다른 사람 폴더는 건드리지 말 것)**: `src/<이름>/`, `notebooks/<이름>/`, `runs/<이름>/` (`<이름>` = `minyeop`, `yeonchang`, `chong`).

- `notebooks/shared/` — `yolov3_refactored.ipynb`(경로를 설정 셀 한 곳에서 관리), `yolov3_20201200_original.ipynb`(원본 실습, 하드코딩 경로라 그대로는 실행 불가). 두 노트북은 옛 흐름 확인용이다.
- `weights/` — Git 제외. `pretrained/`(COCO 사전학습), `legacy/`(기존 가중치 사본: 어떤 이미지로 학습됐는지 몰라 평가에 쓰면 누수 가능).
- `docs/` — `dataset.md`, `troubleshooting.md`, `experiment_results.md`, `evaluation.md`, `strategy.md`, `structure.md`, `original_folder_inventory.md`.
- 원본 `4. X-ray 검사장비 AI 데이터셋/dataset/` — `test1/yolov3/X선이물검출기(06.23_09.22)/`(원본 bmp), `X선이물검출기/`(호기별 재정리 사본, 약 2,500장)가 있다. 탐색 시 제외할 것.

## 데이터 주의 (원본 전수 조사 결과, `docs/original_folder_inventory.md`)

- **파일명 접두(001/002)는 호기가 아니다.** 002 = 1·2호기, 001 = 3호기. 호기는 `data/manifest.csv`의 `unit`을 쓸 것. 라벨된 400장은 1호기 124 / 2호기 109 / 3호기 167이며 해상도가 호기별로 다르다(352x332 / 316x332 / 576x444).
- 원본 bmp 2,809장 중 277장은 같은 호기 내 동일 파일 중복(고유 2,532장). 모두 NG 판정 이미지이며 **정상 이미지는 없다**.
- `.jpg` 서브셋 파일은 실제로 BMP 내용이다.
- `scripts/make_split.py`는 `data/manifest.csv`의 실제 호기로 층화한다(먼저 `scripts/build_manifest.py`, 원본 폴더 필요). 옛 접두 기준 분할은 폐기했고 그 기준의 실험 결과는 쓰지 않는다.

## 데이터/설정 파일 관계

`*.data` → `train`/`valid` 목록 txt + `names` + `classes= 1`. 라벨은 이미지 경로의 `/images/`를 `/labels/`로 바꾼 위치의 같은 이름 `.txt`(YOLO 정규화 포맷)를 자동으로 찾으므로 **이미지 폴더 구조는 `images/`와 `labels/` 형제 폴더를 유지**해야 한다.

- 목록 txt와 `.data`는 **이 PC의 절대경로**를 담고 있어 스크립트/노트북 실행 시 재생성되며 `.gitignore` 대상이다(`data/splits/*.txt|*.data`, `data/samples/train.txt|test.txt|custom.data`). 분할의 원본 기록은 `data/splits/split.csv`.
- 스크립트는 `src/yolov3`를 cwd로 실행하고, 데이터·가중치는 절대경로로 넘긴다.
- 학습 중 검증은 `.data`의 `valid=`를 쓴다. test 평가는 `data/splits/test.data`로 `test.py`를 따로 실행한다.
- 대용량 `.pt`와 `*.bmp`가 많으므로 전체 디렉터리를 무분별하게 Glob/Grep/Read 하지 말 것.

## 구버전 코드 수정 이력 (유지할 것)

최신 PyTorch/numpy/Windows 호환을 위해 `src/yolov3`를 수정했다. 원본으로 되돌리지 말 것(상세: `docs/troubleshooting.md`).

- 모든 `torch.load(...)`에 `weights_only=False`
- `np.int` → `int`
- `train.py`: 이어서 학습 시 `epochs += start_epoch` (기존 `ckpt['epoch']`는 반복 0회 버그)
- `utils/utils.py`: `build_targets`의 `torch.arange(..., device=targets.device)`, `output_to_target`의 `float()` 변환
- `utils/datasets.py`: 한글 경로용 `imread`/`imwrite` 헬퍼(`cv2.imread`는 비ASCII 절대경로를 못 읽음), `detect.py`에서도 사용

## 주요 명령어

저장소 루트에서 `KAMP` 환경으로, **`PYTHONUTF8=1`**(경로에 한글이 있음)을 설정하고 실행한다. 학습·평가는 **표시를 지운 데이터(`--dataset nomark`)**로 한다(표시가 있는 이미지는 모델이 결함이 아니라 표시를 보고 맞힌다, `docs/dataset.md` 8절).

```
conda activate KAMP
set PYTHONUTF8=1
python scripts/check_data.py                      # 데이터 배치 점검
python scripts/make_split.py                      # 분할 (표시 제거본: --src data/nomark --out data/splits_nomark)
python scripts/remove_marks.py build              # 표시 제거 (data/nomark 생성)
python scripts/run_experiment.py --name <내 폴더>/<번호>_<모델>_<설정> --dataset nomark --cfg yolov3-tiny.cfg --weights weights/pretrained/yolov3-tiny.pt --epochs 100 --batch-size 8
python scripts/watch_experiment.py <내 폴더>/<실험> --follow     # 진행 확인
python scripts/evaluate.py <내 폴더>/<실험>        # val에서 임계값 고정 후 test 평가
```

- `--name`의 앞부분이 곧 `runs/` 아래 폴더다. 이미 있는 이름이면 덮어쓰지 않고 중단한다.
- `src/yolov3/train.py`를 직접 쓸 때는 `--data`/`--cfg` 기본값이 COCO용이라 항상 명시하고, 학습 결과는 환경변수 `YOLO_SAVE_DIR`가 없으면 `src/yolov3/weights/last.pt`를 덮어쓴다(과거에 원본 806 epoch 가중치가 덮어써진 적이 있다). `--epochs N`은 기존 체크포인트에서 **추가 N epoch**를 뜻한다.
- 평가는 `evaluate.py`를 쓴다. `test.py`의 P/R/F1은 신뢰도 0.1 한 지점 값이라 쓰지 않는다.
- 노트북 원본 일부 셀의 `—nosave`(긴 대시)는 오타.
