# scripts — 공유 파이프라인

데이터 전처리, 분할, 학습 실행, 평가처럼 **팀이 같이 쓰는 실행 코드**를 두는 폴더입니다. 공유 영역이라 바꾸려면 PR과 팀 확인이 필요합니다. (같은 분할, 같은 평가로 채점해야 모델 비교표가 의미가 있기 때문입니다.)

## 현재 상태: 비어 있음
2026-10-05에 전부 새로 짜기로 해서 파일을 모두 지웠습니다.

## 파일
| 파일 | 역할 |
|---|---|
| `synth_eval.py` | **합성 점 평가 데이터 생성**(시드 42). test 이미지의 실제 점을 지우고, 작대기(막대) 안에만 옅은 점을 합성한다. 팀 모두가 같은 평가 데이터를 만들고 `--verify`로 확인한다 |
| `synth_eval_reference/` | `synth_eval.py --verify`가 비교하는 기준본(점 위치·조건 CSV, 이미지 픽셀 해시). 바꾸면 모든 합성 평가가 비교 불가가 되므로 **함부로 갱신하지 말 것** |

### 합성 점 평가 데이터 (`synth_eval.py`)
```bash
PYTHONUTF8=1 python scripts/synth_eval.py --split test --out data_synth/test --preview   # 생성 + 미리보기 그림 2장
PYTHONUTF8=1 python scripts/synth_eval.py --split test --out data_synth/test --verify    # 기준본과 같은지 확인
```
- 결과 `data_synth/test/`: `images/`, `labels/`(합성 점 박스, 중심 거리 R=5px로 채점 권장), `dots.csv`(점별 위치·진하기·크기·대비), `images.sha256`, `dataset.sha256`, `summary.json`, `skipped.csv`, 미리보기.
- 현재 기준본(test, 시드 42): 이미지 386장, 합성 점 1,436개, 데이터셋 해시 `bc98e02e…b017d0`. 제외된 10장은 막대가 없는 빈 라벨 사진이다.
- 막대 검출 검증: 실제 점 663개 전부가 검출된 막대(9px 확장)에 닿는다. 합성 점은 막대 가장자리에서 3px 이상 안쪽에만 놓는다.
- 평가에 쓰는 모델은 이 이미지에서 검출 결과(JSON)를 만들어 `dots.csv`의 `contrast`, `s`, `f`로 나눠 본다. 해석 시 합성 점은 실제와 약 10%p 차이가 있고 사각 흔적이 있다는 한계를 같이 적는다([docs/analysis.md](../docs/analysis.md)).
- 기준본을 만든 환경: numpy 2.2, opencv-python 5.0, Pillow. 위치·조건이 같고 픽셀만 다르면 라이브러리 버전 차이다.

공통 평가 코드 역할은 임시로 `src/minyeop/faster_rcnn/`의 `metrics.py`(AP·P/R/F1)와 `report.py`(보고 표 계산)가 대신합니다. 모델과 무관한 검출 결과 JSON을 입력으로 받습니다([docs/README.md](../docs/README.md)). 모두가 쓰는 코드로 `scripts/evaluate.py`에 옮길지는 팀 확인이 필요합니다.

## 담당
- **데이터 전처리, 분할**: 이총(`chong`) — 확정 v2 데이터는 `data/`에 있고, 이를 만든 전처리 코드는 main에는 없지만 이총의 브랜치 `feat/chong-preprocess`(`scripts/preprocess/`)에 있습니다(원본 BMP에 돌리면 `data/`와 픽셀 단위로 일치함을 확인함). main에 병합이 필요합니다. 대회 제출물에 전처리 코드가 포함되어야 하므로(재현성 평가) 여기에 커밋해야 합니다. v2 데이터 계보: KAMP 원본 BMP → 중복 제거 → 장비 색상 박스 제거(주변 회색 메움) → 회색조 PNG.
- **학습 실행, 평가**: 팀 합의 후 담당 지정 _(미정)_

## 이전 구현 참고
지운 코드는 Git 이력에 남아 있습니다. 마지막으로 존재한 커밋은 `71813fd`입니다.
```bash
git show 71813fd:scripts/<파일명>.py          # 내용 보기
git checkout 71813fd -- scripts/<파일명>.py   # 복구
```
이전에 있던 파일: `build_manifest`, `check_data`, `check_ring_cue`, `evaluate`, `make_split`, `reeval_experiment`, `remove_marks`, `run_experiment`, `watch_experiment`. (`prepare_n500.py`는 커밋하지 않아 Git에 없습니다.)

## 새로 만들 때 지킬 것
- 한글 경로: 이미지는 `cv2.imdecode(np.fromfile(...))`로 읽고, 설정 파일은 UTF-8로 엽니다. 실행 시 `PYTHONUTF8=1`.
- 신뢰도 임계값은 val에서 정해 고정하고 test는 한 번만 봅니다.
- 실험 결과는 `runs/<이름>/<실험>/`에, 이미 있으면 덮어쓰지 않고 중단합니다.
