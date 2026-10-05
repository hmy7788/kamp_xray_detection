# src/minyeop (허민엽)

허민엽의 모델 코드 폴더입니다. **이 폴더 안은 본인만 수정합니다.** (다른 팀원 폴더, `data/`, `scripts/`, `src/yolov3/`는 PR과 팀 확인이 필요합니다. [CONTRIBUTING.md](../../CONTRIBUTING.md))

## 이 폴더에 둘 것
- 본인이 시도하는 모델/학습 코드 (예: `train.py`, `model.py`)
- 본인 실험의 설정

## 연결되는 다른 폴더
- 탐색용 노트북: `notebooks/minyeop/`
- 실험 결과: `runs/minyeop/<번호>_<모델>_<설정>/` (실행 시 `--name minyeop/<번호>_<모델>_<설정>`)
- 공통 분할은 `data/`의 확정 v2를 그대로 씁니다. 공통 평가 코드(`scripts/`)는 아직 비어 있어 이 폴더의 `metrics.py`, `report.py`가 임시로 대신합니다 (같은 분할, 같은 지표로 비교해야 하므로).
- 모델 공통 결과 표와 기록 규칙: [docs/experiments.md](../../docs/experiments.md), [docs/README.md](../../docs/README.md)

## 담당 모델: Faster R-CNN (`faster_rcnn/`)
torchvision Faster R-CNN ResNet-50 FPN (COCO 사전학습), 결함 1클래스, 입력은 짧은 변 640으로 확대.

| 파일 | 역할 |
|---|---|
| `dataset.py` | `data/<split>`의 YOLO txt를 픽셀 xyxy로 변환, 회색조를 3채널로 복제, 빈 라벨 이미지도 사용, 좌우 반전 |
| `model.py` | 사전학습 모델 생성, 2클래스 헤드, `min_size`·앵커 크기 옵션 |
| `metrics.py` | 검출 결과 기록에서 AP@0.5와 고정 임계값 P/R/F1 계산 |
| `train.py` | 학습 (AMP, epoch마다 val AP로 best 저장, test 미사용, 진행 로그와 `status.json`) |
| `predict.py` | val에서 임계값을 정해 test에 적용, `preds_<split>.json` 저장 |
| `report.py` | 보고 표 계산 (AP·F1 부트스트랩 95% 구간, 완전 검출률, FPPI, 공식/팀 라벨별) |
| `watch.py` | 학습 진행 모니터 (`--follow`로 자동 갱신) |
| `visualize.py` | 검출 그림 (정상 예시, 오류 확대, 빈 라벨, 신뢰도 분포) |

실행 (`PYTHONUTF8=1`을 설정하고 프로젝트 루트에서):
```bash
python src/minyeop/faster_rcnn/train.py   --name 01_frcnn_r50fpn_min640_v1 --epochs 20 --batch-size 4
python src/minyeop/faster_rcnn/watch.py   --name 01_frcnn_r50fpn_min640_v1 --follow      # 다른 터미널에서
python src/minyeop/faster_rcnn/predict.py --name 01_frcnn_r50fpn_min640_v1 --split val  --ckpt best
python src/minyeop/faster_rcnn/predict.py --name 01_frcnn_r50fpn_min640_v1 --split test --ckpt best   # 최종 후보만 한 번
python src/minyeop/faster_rcnn/report.py  --name 01_frcnn_r50fpn_min640_v1 --infer-ms 31
python src/minyeop/faster_rcnn/visualize.py --name 01_frcnn_r50fpn_min640_v1 --split test
```
- 주요 옵션: `--min-size`, `--anchor-sizes 8,16,32,64,128`(작은 앵커, **아직 시험하지 않음**), `--batch-size`, `--epochs`, `--no-amp`, 점검용 `--limit-train`/`--limit-val`
- `predict.py --split test`는 같은 체크포인트(`--ckpt`)로 만든 val 결과(`preds_val.json`)로 임계값을 정합니다. 체크포인트를 바꿨다면 먼저 `--split val`을 다시 실행하세요.
- 실험 폴더가 이미 있으면 `train.py`는 덮어쓰지 않고 중단합니다.
- 사전학습 가중치는 첫 실행 때 자동으로 내려받습니다 (약 160MB). 6GB GPU에서 batch 4, AMP 기준 메모리 약 3.3GB를 썼습니다.
- 결과: `runs/minyeop/01_frcnn_r50fpn_min640_v1/` (test F1 0.989, 공식 라벨만 0.964). 점수 해석 주의는 [docs/README.md](../../docs/README.md)의 "보고할 때의 주의".
