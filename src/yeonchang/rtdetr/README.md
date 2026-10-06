# src/yeonchang/rtdetr — RT-DETR (정연창)

YOLOv3 베이스라인과 비교할 두 번째 모델. ultralytics의 RT-DETR 구현을 쓰고 **COCO 사전학습 `rtdetr-l`**(약 32M 파라미터)에서 시작한다.
`data/` 확정 v2를 그대로 읽으며 분할은 바꾸지 않는다.

## 실행 (PowerShell, 저장소 루트에서)
```powershell
conda activate KAMP
pip install -r src/yeonchang/rtdetr/requirements.txt   # 처음 한 번
$env:PYTHONUTF8="1"
python src/yeonchang/rtdetr/train.py    --exp 02_rtdetrl_img640_e30          # 학습 (약 3시간, epoch당 약 6분, RTX 4050 6GB)
python src/yeonchang/rtdetr/evaluate.py --exp 02_rtdetrl_img640_e30 --split val  # 임계값 결정
python src/yeonchang/rtdetr/evaluate.py --exp 02_rtdetrl_img640_e30 --split test # 최종 1회
```
사전학습 가중치는 처음 실행할 때 `weights/rtdetr-l.pt`로 자동으로 받는다(Git 제외). 긴 학습 중에는 PC 절전을 끈다.

## 학습 설정 (`train.py`의 `HYP`, 인자로 바꿀 수 있음)
| 항목 | 값 | 이유 |
|---|---|---|
| imgsz | 640 | 원본 최대 576x444. 약 10px 결함을 조금이라도 크게 |
| epochs / patience | 30 / 10 | 1클래스·COCO 사전학습 미세조정. 72 epoch는 epoch당 약 6분이라 너무 길어 줄임(01은 3 epoch 후 중단, `_incomplete_`) |
| batch / nbs | 4 / 16 | 6GB GPU, 기울기 누적 4회 |
| optimizer | AdamW, lr0 1e-4, wd 1e-4, 코사인, warmup 2 | DETR 계열 표준 |
| 증강 | 좌우·상하 뒤집기, 밝기 ±0.3, 이동 0.1, 축소 0.2 | 회색조라 색상 증강 끔. mosaic·큰 축소는 결함을 더 작게 만들어 끔 |

## 평가 (`evaluate.py`)
- 예측은 신뢰도 0.001 이상 모두 저장 → **val에서 F1 최대 임계값을 정해 고정** → test에 그대로 적용, test는 한 번만(이미 채점했으면 중단).
- 매칭: 신뢰도 순 탐욕 매칭, IoU ≥ 0.5, 정답 하나에 예측 하나. AP@0.5는 VOC 전 구간 보간. 보조로 중심 거리 5px 이내 매칭도 낸다.
- 보고: P/R/F1, TP/FP/FN, AP@0.5, FPPI, 이미지 검출률·완전 검출률, 빈 라벨 이미지의 FP, 출처(공식/팀)·호기·해상도·월별 분해, 추론 ms/장.
- `preds_<split>.json`의 `gt_match`(image_id, box_idx, 매칭 신뢰도)는 `data/conditions.csv`와 조인해 조건별 미탐지 분석에 쓴다.
- `test.py`의 P/R/F1(신뢰도 0.1 한 지점)은 쓰지 않는다. 학습 중 ultralytics가 출력하는 mAP도 참고값이다.

## 산출물 `runs/yeonchang/<실험>/`
`config.json`, `train.log`, `args.yaml`, `results.csv`, 학습 곡선 그림, `eval_report_{val,test}.json`은 Git에 올린다(로컬 경로는 `<KAMP>`/`<HOME>`으로 자동 치환).
`weights/`(가중치, 생성된 data.yaml), `preds_*.json`은 `.gitignore`로 제외된다. 끝까지 돌지 못한 실험은 `_incomplete_`로 이름을 바꾼다.
