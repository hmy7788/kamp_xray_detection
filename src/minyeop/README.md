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

## 담당 모델: YOLOv3-tiny (`yolov3_tiny/`)
공유 코드 `src/yolov3`를 수정하지 않고 실험 폴더 안에서 `train.py`를 실행하는 껍데기와 예측·모니터입니다. COCO 사전학습에서 시작하고 `valid`는 val만 씁니다(test는 학습에 쓰지 않음).

| 파일 | 역할 |
|---|---|
| `run.py` | 실험 폴더 `runs/minyeop/<name>/`을 만들고 `kamp.data`·`config.json`(상대 경로만)을 쓴 뒤 `src/yolov3/train.py` 실행. 로그는 `train.log` |
| `predict.py` | val/test 추론 후 Faster R-CNN과 같은 형식의 `preds_<split>.json` 저장 (신뢰도 0.001 이상, NMS IoU 0.6) |
| `watch.py` | 진행 모니터 (`results.txt`와 로그를 읽음) |

```bash
python src/minyeop/yolov3_tiny/run.py     --name 02_yolov3tiny_img640_v1 --epochs 100 --batch-size 16
python src/minyeop/yolov3_tiny/watch.py   --name 02_yolov3tiny_img640_v1 --follow
python src/minyeop/yolov3_tiny/predict.py --name 02_yolov3tiny_img640_v1 --split val  --ckpt best
python src/minyeop/yolov3_tiny/predict.py --name 02_yolov3tiny_img640_v1 --split test --ckpt best   # 한 번만
python src/minyeop/faster_rcnn/report.py  --name 02_yolov3tiny_img640_v1    # 공통 보고 표 (--center-r 5 --tag _v2 로 평가 v2)
```
- 결과: test F1 0.989(공식 라벨만 0.964), 임계값 0.06, 학습 59분. 증강은 모자이크·색상·좌우 반전(크기·회전 증강은 0).
- `best.pt`는 학습 코드가 val의 0.99·mAP@0.5+0.01·R로 고른 epoch이다. `train.log`에는 개인 PC 경로가 들어가므로 Git에 올리지 않는다.

## Faster R-CNN MobileNetV3-FPN (`faster_rcnn_mobilenet/`)
`faster_rcnn/`의 학습·예측 코드를 그대로 쓰고 모델 생성 함수만 `fasterrcnn_mobilenet_v3_large_fpn`(COCO 사전학습)으로 바꿔 끼웁니다. 같은 조건으로 ResNet-50 버전과 비교하려는 것이며, 이 FPN은 stride 16 이상의 특징맵만 써서 작은 결함에는 불리할 것으로 봤지만, 실제로는 찾는 능력이 ResNet-50과 같은 수준이었고 박스 정밀도(mAP50-95 0.559 대 0.611)만 낮았습니다.
```bash
python src/minyeop/faster_rcnn_mobilenet/train.py   --name 04_frcnn_mobv3_min640_anc16_v1 --epochs 20 --batch-size 4 --anchor-sizes 16,32,64,128,256
python src/minyeop/faster_rcnn_mobilenet/predict.py --name 04_frcnn_mobv3_min640_anc16_v1 --split val --ckpt best
```
- 결과: test F1 0.986(공식 라벨만 0.959), 임계값 0.93, 학습 20분, 파라미터 19.4M. 앵커는 16~256 한 가지만 시험했다(기본 앵커와 비교하지 않음).

## 분석: 점 제거 (`dot_removal/`)
`remove_dot.py`: test 이미지의 결함(어두운 점)을 보간으로 지운 이미지를 임시 폴더에 만들어 두 모델로 다시 추론하고, 같은 자리에서 검출이 남는지 센다. 원본 `data/`는 바꾸지 않는다.
```bash
python src/minyeop/dot_removal/remove_dot.py --work-dir <임시 폴더> --out runs/minyeop/03_dot_removal_v1/ns_half6 --half 6 --method ns
python src/minyeop/dot_removal/remove_dot.py --work-dir <임시 폴더> --out runs/minyeop/03_dot_removal_v1/mean_half6 --half 6 --method mean --noise-gain 0
```
`--half`는 제거 정사각형의 반변(6 → 13x13px), `--method`는 `ns`(Navier-Stokes) 또는 `mean`(주변 평균), `--noise-gain`은 메운 자리에 더할 노이즈 세기. 결과 해석은 [docs/README.md](../../docs/README.md)의 "점 제거 실험".
