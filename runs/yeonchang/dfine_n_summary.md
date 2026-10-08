# D-FINE-N 실험 요약

작성: 정연창 (2026-10-08). 실험 폴더: `runs/yeonchang/03_dfinen_img640_e50/`. 데이터: 확정 v2.

## 한 줄 요약
파라미터 3.7M인 D-FINE-N이 test F1 0.989, AP@0.5 0.989로 RT-DETR-l(32.8M)과 정확도가 같고, CPU 4스레드에서 장당 97ms(초당 10.3장)로 3.6배 빠르다. test에서 실제로 놓친 결함은 0건이고, IoU 기준 미검출 7건은 모두 공식 라벨과 박스 위치·크기가 어긋난 경우다.

## 왜 D-FINE-N인가
- COCO 사전학습 가중치가 공개된 DETR 계열 중 가장 작다. HuggingFace에서 확인한 파라미터 수: D-FINE-N 3.8M(1클래스 헤드로 바꾸면 3.72M), D-FINE-S 10.4M, RT-DETRv2-R18 20.2M, RT-DETR-l 32.8M.
- RT-DETR 구조를 바탕으로 박스 위치 예측을 개선한 모델이라, 같은 계열에서 크기만 줄였을 때를 비교할 수 있다.
- HuggingFace `transformers`가 공식 지원한다(`ustc-community/dfine-nano-coco`, Apache-2.0). 원저자 저장소를 따로 받을 필요가 없다.

## 학습 설정
| 항목 | 값 |
|---|---|
| 구현 | transformers 5.18.0 (`DFineForObjectDetection`), torch 2.6.0+cu124 |
| 시작 가중치 | `ustc-community/dfine-nano-coco` (COCO). 클래스 헤드만 1클래스로 새로 초기화 |
| 입력 | 640×640으로 늘림(비율 유지 안 함), /255, 정규화 없음 (체크포인트 기본값) |
| epochs / patience | 50 / 10 → **43 epoch에서 조기 종료, best = 33 epoch** |
| batch | 16 |
| optimizer | AdamW, lr 2e-4 (백본 1e-4), weight decay 1e-4 (bias·정규화 층 제외) |
| 스케줄 | warmup 1 epoch, 코사인으로 0.1배까지, grad clip 0.1 |
| EMA / AMP | 0.9999 (초반 2000회 완만) / FP16 혼합 정밀도 |
| 증강 | 밝기 ±0.3, 좌우·상하 뒤집기 0.5, 이동 0.1, 크기 ±0.2. 색상 증강·mosaic 없음 (RT-DETR-l과 같은 방향) |
| best 선택 기준 | val 0.1×AP50 + 0.9×AP50-95 (EMA 가중치, RT-DETR-l과 같은 기준) |
| seed | 0 |
| 학습 시간 | 실제 97.9분 (epoch당 약 2.4분). 36 epoch에서 노트북 절전으로 935.7분 멈췄다가 재개, `config.json`의 `pause`에 기록 |

- 결함이 약 10px로 작아서, 결함을 더 작게 만드는 mosaic과 큰 축소는 쓰지 않았다.
- val AP@0.5는 1 epoch에 0.976, 8 epoch 이후 0.98~0.99에 머물렀다. AP@0.5:0.95는 33 epoch에서 0.653으로 가장 높았다.

## 정확도
신뢰도 임계값은 val F1 최대로 **0.7114**에 고정했고, test는 그 값으로 한 번만 채점했다. 매칭은 신뢰도 순 탐욕 매칭(IoU ≥ 0.5, 1:1)이고, AP는 VOC 전 구간 보간이다.

| 지표 | val (369장, 정답 606) | test (396장, 정답 663) |
|---|---|---|
| AP@0.5 | 0.988 | **0.989** |
| Precision / Recall | 0.989 / 0.990 | 0.988 / 0.989 |
| F1 | 0.989 | **0.989** |
| TP / FP / FN | 600 / 7 / 6 | 656 / 8 / 7 |
| FPPI (이미지당 오검출) | 0.019 | 0.020 |
| 이미지 검출률 / 완전 검출률 | 0.997 / 0.983 | 1.000 / 0.981 |
| 빈 라벨 이미지 중 오검출 | 0 / 19 | 0 / 27 |
| mAP@0.5:0.95 | − | 0.655 |

### test 조건별 F1 (괄호: TP / FP / FN)
| 구분 | 값 |
|---|---|
| 라벨 출처 | 공식 0.961 (186/8/7), 팀 **1.000** (470/0/0) |
| 호기 | 1호기 0.975 (178/5/4), 2호기 0.991 (224/2/2), 3호기 0.996 (254/1/1) |
| 해상도 | 316×332 0.993, **352×332 0.962**, 412×332 1.000, 576×444 0.996 |
| 촬영 월 | **6월 0.965**, 7·8·9월 1.000 |

오류는 공식 라벨, 1호기 352×332, 6월에 몰려 있다. FP 8건·FN 7건이 모두 공식 라벨 이미지에서 나왔으므로, 나머지 두 조건은 공식 라벨 이미지가 그 구간에 많아서 겹쳐 보이는 것으로 추정한다(구간별 공식 라벨 비율은 확인하지 않음).

### 중심 거리 기준 (보조 지표, test, 임계값 0.7114)
| 기준 | F1 | TP / FP / FN |
|---|---|---|
| IoU 0.5 | 0.989 | 656 / 8 / 7 |
| 중심 2px 이내 | 0.987 | 655 / 9 / 8 |
| 중심 3px 이내 | 0.998 | 662 / 2 / 1 |
| 중심 5px 이내 | 0.999 | 663 / 1 / 0 |

- 중심 3px 기준의 FN 1건 `001_20200625_043337(3)` #0(공식)은 예측이 라벨 중심에서 3.14px라 기준을 0.14px 넘긴 경우다. IoU 기준으로는 TP다.
- 중심 5px에서는 FN이 0이다.

## 오류분석: 미검출 7건은 모두 라벨과의 차이
모델은 7건 모두에서 결함 점의 중심을 0.48~0.77px 이내로 찾았다. 실제 점 위치는 이미지에서 black top-hat으로 자동 검출했다.

| 이미지 #박스 | 라벨 크기 | 라벨↔점 | 예측↔점 | IoU | 라벨을 점 중심으로 옮긴 IoU | 원인 |
|---|---|---|---|---|---|---|
| 002_20200624_123122(6) #2 | 7×11 | **2.68px** | 0.65px | 0.469 | **0.669** | 라벨 치우침 |
| 002_20200624_003132(0) #1 | 13×14 | **3.23px** | 0.70px | 0.486 | **0.534** | 라벨 치우침 |
| 002_20200623_043039(3) #2 | 6×7 | 0.77px | 0.48px | 0.451 | 0.451 | 라벨이 작음 |
| 002_20200623_203040(4) #1 | 9×5 | 1.40px | 0.62px | 0.428 | 0.461 | 라벨이 작음(납작) |
| 002_20200624_123126(4) #1 | 6×7 | 0.54px | 0.59px | 0.440 | 0.440 | 라벨이 작음 |
| 002_20200624_003136(5) #0 | 14×15 | 1.17px | 0.57px | 0.463 | 0.463 | 라벨이 큼 |
| 001_20200625_043346(2) #1 | 11×8 | 1.60px | 0.77px | 0.486 | 0.486 | 라벨이 작음(납작) |

- 모델은 결함 크기와 무관하게 약 10px(3호기 약 13px) 박스를 그린다. 학습 라벨의 75%가 클릭 중심에 고정 크기 네모를 만든 팀 라벨이라 크기를 배울 정보가 없다.
- 공식 라벨은 대부분 점 위에 있지만(라벨↔점 중앙값 0.99px), 2px 이상 치우친 박스가 6.9%로 팀 라벨(1.0%)보다 많고, 3호기는 16.3%다. 자세한 내용은 `runs/yeonchang/label_offset_analysis.md`.
- **RT-DETR-l이 놓친 저대비 결함** `001_20200630_043548(7)` #0(국소 대비 0.07)을 D-FINE-N은 신뢰도 0.855로 검출했다. RT-DETR-l은 0.563으로 임계값 0.8271 아래였다.
- test 라벨을 고쳐 다시 채점하지는 않았다. `data/`는 공유 영역이고, 결과를 본 뒤 정답을 바꾸면 평가가 오염된다.

## 속도
batch 1, 장당 처리 시간(전처리 640 리사이즈 + 추론 + 후처리, 이미지 파일 읽기 제외)의 중앙값이다. 측정 환경은 RTX 4050 Laptop / i7-13620H, AC 전원, Windows 전원 모드 "최고 성능", val 100장, 예열 10회, RT-DETR-l과 번갈아 3회 반복한 중앙값의 중앙값이다.

| 환경 | D-FINE-N | (참고) RT-DETR-l | 회차별 중앙값 (D-FINE-N) |
|---|---|---|---|
| GPU FP32 | **32 ms (30.9 장/초)** | 33 ms | 32.3 / 22.6 / 33.5 |
| GPU FP16 (`model.half()`) | **30 ms (33.4 장/초)** | 27 ms | 23.5 / 30.0 / 30.3 |
| CPU 4스레드 (코어 0·2·4·6 고정) | **97 ms (10.3 장/초)** | 353 ms | 119.4 / 96.9 / 93.0 |
| CPU 1스레드 (코어 0 고정) | **172 ms (5.8 장/초)** | 1,067 ms | 172.1 / 171.6 / 167.6 |

- **CPU에서는 RT-DETR-l보다 4스레드 3.6배, 1스레드 6.2배 빠르다.** 연산량 차이(D-FINE-N 약 7 GFLOPs는 논문 값으로 기억에 의존, RT-DETR-l 109.9 GFLOPs는 학습 로그)가 그대로 드러난다.
- **GPU batch 1에서는 RT-DETR-l과 차이가 없다.** transformers 구현은 박스 정제 과정에 작은 연산이 수천 개라 호출 오버헤드가 지배한다. 프로파일러로 보면 reshape 261회, mul 208회, add 197회 등이다. 여러 장을 묶으면(batch 8) 장당 시간이 5배 넘게 줄어든다("균형 조정" 모드 측정). ONNX Runtime이나 TensorRT로 연산을 합치면 더 빨라질 여지가 있다(미검증).
- **FP16은 거의 효과가 없다.** autocast 방식은 형 변환이 늘어 오히려 느려서 쓰지 않았다. FP16 정확도는 확인하지 않았다.
- **현장 가정**: 촬영 묶음 안 연속 사진 간격은 중앙값 4초, 최소 1초다(초 단위 기록, 불량 사진만 저장). GPU 없는 산업용 PC(CPU 4스레드)에서 장당 0.10초로 1초 대비 10배 여유가 있다. CPU 1스레드로 제한돼도 0.17초로 충족한다.

### 측정에서 겪은 함정 (다시 잴 때 주의)
- 배터리 모드에서 잰 값은 무효다(`speed_benchmark_invalid_battery.json`).
- "균형 조정" 모드에서는 GPU가 83~115ms로 약 3배 느리게 나왔다(`speed_benchmark.json`, 사용 안 함).
- i7-13620H는 0~11번이 고성능 코어, 12~15번이 저전력 코어(같은 계산이 3~4배 느림)다. CPU 측정은 고성능 코어에 고정해야 한다.

## 다른 모델 대비 위치
- RT-DETR-l과 비교하면 정확도는 오차 범위 안에서 같다(test TP 656 vs 655). 부트스트랩 신뢰구간은 계산하지 않았다.
- 두 모델 모두 공식 라벨 F1 0.961로, 같은 라벨 문제에서 같은 오류를 낸다. 모델 크기와 무관하다.
- 학습 조건은 다르다. epoch(43 vs 30), 전처리(늘림 vs 비율 유지+패딩), 프레임워크(transformers vs ultralytics). 비교할 때 함께 적어야 한다.

## 파일과 재현
| 파일 | 내용 |
|---|---|
| `src/yeonchang/dfine/train.py` | 학습. 실험 폴더가 있으면 중단, 데이터 버전 확인, 로컬 경로 치환 |
| `src/yeonchang/dfine/predict.py` | val/test 예측을 `preds_<split>.json`으로 저장 (test는 val 채점 뒤 한 번만) |
| `src/yeonchang/rtdetr/evaluate.py` | 공통 채점 (`--reuse-preds`로 저장된 예측 채점) |
| `src/yeonchang/dfine/speed.py` | 속도 측정 (`--gpu-repeat`, `--cpu-repeat`, 코어 고정, 전원 상태 기록) |
| `src/yeonchang/rtdetr/center_sensitivity.py`, `label_offset.py` | 중심 거리 민감도, 라벨 치우침 분석 |
| `runs/yeonchang/03_dfinen_img640_e50/` | `config.json`, `train.log`, `results.csv`, `eval_report_{val,test}.json`, `eval_center_sensitivity.json`, `speed_gpu_repeat.json`, `speed_cpu_pinned.json` |
| `…/weights/best/` | 가중치 폴더(`model.safetensors` 15MB + 설정). `best.pt` 파일 대신 이 폴더 형식이다. Git 제외 |

```powershell
conda activate KAMP
pip install transformers==5.18.0
$env:PYTHONUTF8="1"
python src/yeonchang/dfine/train.py --exp 03_dfinen_img640_e50
python src/yeonchang/dfine/predict.py --exp 03_dfinen_img640_e50 --split val
python src/yeonchang/rtdetr/evaluate.py --exp 03_dfinen_img640_e50 --split val --reuse-preds
python src/yeonchang/dfine/predict.py --exp 03_dfinen_img640_e50 --split test
python src/yeonchang/rtdetr/evaluate.py --exp 03_dfinen_img640_e50 --split test --reuse-preds
python src/yeonchang/rtdetr/center_sensitivity.py --exp 03_dfinen_img640_e50
python src/yeonchang/dfine/speed.py --exp 02_rtdetrl_img640_e30,03_dfinen_img640_e50 --gpu-repeat 3 --n 100 --warmup 10
python src/yeonchang/dfine/speed.py --exp 02_rtdetrl_img640_e30,03_dfinen_img640_e50 --cpu-repeat 3 --n 100 --warmup 10
```
`transformers` 5.18.0을 설치해도 torch·numpy·opencv 버전은 바뀌지 않는다(설치 전 dry-run으로 확인).

## 한계와 남은 일
- FP16(`model.half()`)의 정확도는 확인하지 않았다. 운영 옵션으로 쓰려면 val에서 먼저 확인해야 한다.
- 점 지우기 실험(모델이 위치를 외웠는지 확인)은 RT-DETR-l로만 했다. D-FINE-N은 하지 않았다.
- 속도는 노트북 한 대의 값이다. 실제 산업용 PC에서는 달라진다.
- 같은 제품·장비 3대 안의 결과다. 다른 제품이나 다른 형태의 이물로도 통하는지는 이 데이터로 알 수 없다.
- 코드와 03 결과는 아직 커밋하지 않았다(브랜치 `feat/yeonchang-rtdetr`).
