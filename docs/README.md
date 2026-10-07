# docs — 문서

데이터, 실험, 평가 기준 같은 **프로젝트 문서**를 두는 폴더입니다.

## 현재 문서
| 문서 | 내용 |
|---|---|
| [metrics.md](metrics.md) | 검출 모델의 성능 지표 정리 (박스·이미지 단위, 조건별 분해, 신뢰도, 불확실성, 주의사항) |
| [experiments.md](experiments.md) | 모델 공통 성능 지표 표만 모은 문서: val·test 결과, 라벨 출처별 결과 (Faster R-CNN 기록 포함) |
| [analysis.md](analysis.md) | 추가 분석의 해석과 한계: 호기·해상도별 통계, 가짜 정상 오경보, 점 합성 시험, 추론 속도, 모델 선정 기준(제안), test 사용 이력 |

2026-10-05에 이전 문서 7개를 지웠고, 새로 작성하는 문서를 이 폴더에 둡니다.

## 실험 지표 기록 가이드
[experiments.md](experiments.md)의 표를 채울 때의 규칙입니다. [metrics.md](metrics.md)의 "핵심 지표"와 "최종 보고 표 양식"을 실제로 채우는 방법입니다.
- 마지막 갱신: 2026-10-05
- 담당자는 **본인 모델의 행만** 수정합니다 (PR 충돌 방지). 표의 열은 바꾸지 마세요 (바꾸면 모델끼리 비교 불가)

### 1. 채우기 전에 지킬 규칙 (모든 모델 공통)
| 항목 | 규칙 |
|---|---|
| 데이터 | `data/` 확정 v2. `manifest.csv`의 sha256이 `1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623`인지 확인 |
| 분할 | train 1,767 / val 369 / test 396 그대로 사용. **다시 나누지 말 것** |
| 시작 가중치 | COCO 사전학습본 (이전 학습본으로 시작하면 평가에 누수) |
| 체크포인트 선택 | **val로만** 고른다 (예: val AP가 최대인 epoch). 고르는 기준은 test를 보기 전에 정해 둔다 |
| 신뢰도 임계값 | **val에서 F1이 최대인 값**을 정해 test에 그대로 적용. test에서 다시 정하지 않음 |
| test 평가 | 최종 후보 **한 번만**. test 결과를 보고 모델·epoch·임계값·설정을 바꾸지 않음 |
| 매칭 | IoU 0.5, 신뢰도가 높은 검출부터 아직 짝이 없는 정답과 짝지음 (TP), 짝이 없으면 FP, 남은 정답은 FN |
| AP | 신뢰도 0.001 이상 전체 검출로 PR 곡선, 전 구간 보간(VOC 방식) |
| 라벨 출처 | 결과를 **전체 / 공식 라벨 / 팀 라벨**로 나눠 함께 보고 (3절의 이유 참고) |
| 모델 자체 지표 | `test.py` 같은 도구가 내는 P/R/F1은 쓰지 않음 (신뢰도 0.1 한 지점 값). 아래 계산기로 통일 |

### 2. 채우는 순서 (예: 새 모델)
1. **학습**: 결과를 `runs/<본인이름>/<실험이름>/`에 둔다 (예: `runs/yeonchang/01_yolov8s/`). 실험 이름은 `<번호>_<모델>_<핵심설정>` 형식
2. **검출 결과 기록 저장**: val과 test를 추론해 아래 형식의 `preds_val.json`, `preds_test.json`을 실험 폴더에 만든다
3. **계산기 실행**: 표의 숫자를 한 번에 계산
   ```bash
   # PYTHONUTF8=1 을 설정하고 실행
   python src/minyeop/faster_rcnn/report.py \
     --preds-val  runs/<이름>/<실험>/preds_val.json \
     --preds-test runs/<이름>/<실험>/preds_test.json --infer-ms <ms/장>
   ```
   - 출력된 마크다운 표 줄을 [experiments.md](experiments.md)의 두 표에 붙여 넣는다 ("전체" 줄은 첫 표, "공식 라벨"·"팀 라벨" 줄은 둘째 표)
   - 임계값은 val에서 자동으로 정해진다 (`--threshold`로 직접 지정할 수도 있음)
   - `report_val.json`, `report_test.json`이 `preds_*.json`과 같은 폴더(`--name` 사용 시)에 저장된다
4. **설정 기록**: 아래 "실험 설정 기록" 표에 학습 설정을 적는다
5. PR에는 본인 행과 실험 폴더(`config`, 로그, `results.csv`, `report_*.json`)만 포함

> 계산기(`report.py`)와 지표 함수(`metrics.py`)는 현재 `src/minyeop/faster_rcnn/`(개인 영역)에 있다. 다른 사람이 **읽거나 실행만** 하고 수정하지 말 것. 모두가 쓰는 공통 코드는 `scripts/evaluate.py`로 옮기는 것이 맞지만 공유 영역이라 팀 확인 후 PR이 필요하다 (미정).

#### 검출 결과 기록 형식 (`preds_<split>.json`)
이미지마다 한 항목인 JSON 배열. **모델과 무관**하다. 좌표는 모두 **원본 이미지 픽셀 기준** `x1, y1, x2, y2`.
```json
[{"name": "002_20200622_203053(2).png", "w": 352, "h": 332,
  "gt":   [[x1, y1, x2, y2], ...],
  "dets": [[x1, y1, x2, y2, score], ...]}]
```
- `name`: `data/<split>/images/`의 파일명 (`manifest.csv`의 `image_id` + `.png`)
- `gt`: 라벨 txt(`0 cx cy w h`, 0~1 비율)를 픽셀 좌표로 바꾼 것. 빈 라벨 이미지는 `[]`
- `dets`: **신뢰도 0.001 이상 검출을 모두** 저장 (임계값으로 미리 자르면 AP가 틀리고 임계값을 고를 수 없음). 모델의 최대 검출 수 제한은 이미지당 100개 정도
- 모든 이미지(val 369장, test 396장)를 빠짐없이 포함 (빈 라벨 이미지도)
- 정답 좌표가 들어 있으므로 `.gitignore`에 의해 Git에는 올라가지 않음

### 3. 보고할 때의 주의 (이 데이터의 특성)
- **팀 라벨은 박스 크기가 고정**(1·2호기 10px, 3호기 약 13px)이라 모델이 그 크기만 맞히면 IoU가 쉽게 나온다. 이 때문에 "전체" 수치가 부풀려진다 (Faster R-CNN: 팀 라벨 F1 1.000 vs 공식 라벨 0.964). **공식 라벨 줄이 크기까지 맞히는 실제 난이도**에 가깝다
- 오류 대부분이 "놓침"이 아니라 "박스 크기 차이로 IoU 0.5 미달"이다 (Faster R-CNN test: 오류 7건 모두 같은 점을 찾았고 중심 거리 1.2px 이내). 수치가 모델 간 차이를 가리지 못할 수 있으니 **전체 점수만으로 우열을 결론 내리지 말 것**
- 신뢰구간은 이미지를 독립으로 보고 계산한 값이다. 연속 촬영본은 서로 닮아서 실제 불확실성은 더 크다
- 완전 검출률은 "그 이미지의 결함을 전부 찾은 비율"(FN 0), FPPI는 이미지당 평균 FP 수 (결함 이미지 기준)
- 빈 라벨 이미지(val 19장, test 27장)는 제품이 화면에 반만 나오는 특수한 사진이라 정상 제품의 오경보율로 해석하지 말 것
- 결함이 막대(테스트피스) 끝에 있는 규칙성과 표시를 지운 흔적을 모델이 지름길로 쓰는지는 **아직 검증하지 않았다**. 높은 점수를 일반 성능으로 해석하지 말 것

### 표 읽는 법 ([experiments.md](experiments.md))
- 임계값은 모델마다 val에서 정한 값. 괄호는 이미지 단위 부트스트랩 95% 구간(1,000회)
- 추론 시간은 RTX 4050 Laptop 6GB, 혼합 정밀도(AMP), 배치 4, 전처리·후처리 포함 평균 (측정 조건이 다르면 모델끼리 비교하지 말 것)
- 모델 간 차이는 구간이 겹치는지 함께 서술 (구간이 0을 포함하면 "우열 확정 불가"로 기술)
- 라벨 출처별 표: 공식 라벨 = KAMP 제공(500장, 크기가 실제 결함 반영), 팀 라벨 = 팀이 만든 2,032장(고정 크기). 평가셋 안의 구성은 val 공식 59 / 팀 310, test 공식 85 / 팀 311. 구간은 "전체" 줄에만 계산됨

### 실험 설정 기록
재현에 필요한 항목. 학습 코드가 저장하는 `config.json`이 있으면 그것을 근거로 적는다. 본인 행만 추가한다.

| 실험 이름 | 담당 | 브랜치 | 모델 / 사전학습 | 입력 크기 | 배치 | epoch (선택된 epoch) | 옵티마이저 / lr | 증강 | 시드 | 학습 시간 | 가중치 크기 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `01_frcnn_r50fpn_min640_v1` | minyeop | `feat/minyeop-faster-rcnn` | torchvision Faster R-CNN ResNet-50 FPN / COCO | 짧은 변 640 (최대 1000), 기본 앵커 32~512 | 4 | 20 (**7**, val AP 최대) | SGD(모멘텀 0.9, wd 5e-4) lr 0.005, 워밍업 후 코사인 | 좌우 반전 | 0 | 1시간 07분 (RTX 4050 6GB, AMP) | 166MB | 빈 라벨 이미지도 학습에 사용 |
| `02_yolov3tiny_img640_v1` | minyeop | `feat/minyeop-faster-rcnn` | ultralytics YOLOv3-tiny(2020) / COCO | 640 고정 | 16 (누적 4 → 유효 64) | 100 (**59**번째, best.pt 기준 0.99·mAP@0.5+0.01·R) | SGD(모멘텀 0.937) lr 0.01에서 코사인 감소, 기본 하이퍼파라미터 | 모자이크, 색상(HSV), 좌우 반전 (크기·회전 증강은 0) | 0 (코드 기본값) | 59분 (RTX 4050 6GB, AMP 미사용) | 69MB | 추론은 배치 1·fp32라 Faster R-CNN과 시간 직접 비교 불가 |
| `04_frcnn_mobv3_min640_anc16_v1` | minyeop | `feat/minyeop-faster-rcnn` | torchvision Faster R-CNN MobileNetV3-Large FPN / COCO | 짧은 변 640 (최대 1000), 앵커 16·32·64·128·256 (3개 레벨 공통) | 4 | 20 (**18**, val AP 최대) | SGD(모멘텀 0.9, wd 5e-4) lr 0.005, 워밍업 후 코사인 | 좌우 반전 | 0 | 20분 (RTX 4050 6GB, AMP) | 76MB | 기본 앵커와 비교하지 않음 |
| | | | | | | | | | | | | |

재현 명령 (Faster R-CNN):
```bash
PYTHONUTF8=1 python src/minyeop/faster_rcnn/train.py --name 01_frcnn_r50fpn_min640_v1 --epochs 20 --batch-size 4
PYTHONUTF8=1 python src/minyeop/faster_rcnn/predict.py --name 01_frcnn_r50fpn_min640_v1 --split val  --ckpt best
PYTHONUTF8=1 python src/minyeop/faster_rcnn/predict.py --name 01_frcnn_r50fpn_min640_v1 --split test --ckpt best   # 한 번만
PYTHONUTF8=1 python src/minyeop/faster_rcnn/report.py  --name 01_frcnn_r50fpn_min640_v1 --infer-ms 31
```

재현 명령 (YOLOv3-tiny):
```bash
PYTHONUTF8=1 python src/minyeop/yolov3_tiny/run.py --name 02_yolov3tiny_img640_v1 --epochs 100 --batch-size 16
PYTHONUTF8=1 python src/minyeop/yolov3_tiny/predict.py --name 02_yolov3tiny_img640_v1 --split val  --ckpt best
PYTHONUTF8=1 python src/minyeop/yolov3_tiny/predict.py --name 02_yolov3tiny_img640_v1 --split test --ckpt best   # 한 번만
PYTHONUTF8=1 python src/minyeop/faster_rcnn/report.py  --name 02_yolov3tiny_img640_v1
```

### 두 모델 비교에서 확인한 것 (2026-10-06)
- **임계값 눈금이 모델마다 크게 다르다**: Faster R-CNN 0.95, YOLOv3-tiny 0.06. 같은 임계값을 팀 전체에 강제하면 안 되는 구체적 근거다.
- **test 오류가 같다**: 두 모델 모두 TP 656 / FP 7 / FN 7이고, Faster R-CNN의 오류 이미지 5장은 YOLOv3-tiny 오류 이미지 6장에 모두 포함된다. 모델이 달라도 같은 이미지에서 틀리므로 모델 약점이 아니라 **라벨(정답 박스 크기)과 IoU 0.5 기준의 문제**일 가능성이 높다.
- 그래서 IoU 0.5 기준 F1로는 두 모델을 구별할 수 없다. AP@0.5는 test 0.988 대 0.981(구간이 크게 겹침)이라 우열을 확정할 수 없다.
- 두 모델의 학습 조건이 다르다(증강, AMP, 배치, 학습 이미지 처리). 이 차이를 보고서에 함께 적을 것.

### 세 번째 모델: Faster R-CNN MobileNetV3-FPN (2026-10-06)
```bash
PYTHONUTF8=1 python src/minyeop/faster_rcnn_mobilenet/train.py   --name 04_frcnn_mobv3_min640_anc16_v1 --epochs 20 --batch-size 4 --anchor-sizes 16,32,64,128,256
PYTHONUTF8=1 python src/minyeop/faster_rcnn_mobilenet/predict.py --name 04_frcnn_mobv3_min640_anc16_v1 --split val  --ckpt best
PYTHONUTF8=1 python src/minyeop/faster_rcnn_mobilenet/predict.py --name 04_frcnn_mobv3_min640_anc16_v1 --split test --ckpt best   # 한 번만
PYTHONUTF8=1 python src/minyeop/faster_rcnn/report.py --name 04_frcnn_mobv3_min640_anc16_v1 [--center-r 5 --tag _v2]
```
- 결과(test): F1 0.986, AP 0.985, mAP50-95 0.559, 공식 라벨 F1 0.959, 임계값 0.93. **찾는 능력은 ResNet-50·YOLOv3-tiny와 같은 수준**(평가 v2에서 663개 중 662개)이고, **박스 정밀도는 ResNet-50보다 낮다**(mAP50-95 0.559 대 0.611, R=2px F1 0.965 대 0.988). 이 FPN이 낮은 해상도 특징맵(stride 16 이상)만 쓰는 약점이 위치 정밀도에서만 드러났다.
- ResNet-50 대비 파라미터 19.0M(41.4M), 학습 20분(67분), 같은 조건(배치 4, AMP)에서 추론 약 4배 빠름(8ms 대 31ms).
- 오류 9건(FP 9 / FN 9): 공식 라벨 7장(8건)은 박스 크기 차이, 팀 라벨 2장은 중복 검출 1건과 신뢰도 0.910이라 임계값 0.93에 못 미친 1건이다. 진짜로 결함을 놓친 것은 없다.
- 한계: 앵커 16~256 설정 하나만 시험했고(기본 앵커와 비교 안 함), 시드 하나, 모델마다 학습 조건(증강, AMP, 배치, epoch)이 달라 우열을 단정할 수 없다.

### 평가 v2(중심 거리 기준)와 점 제거 실험 (2026-10-06, 제안·분석 단계)
- **평가 v2**: IoU 대신 "검출 중심이 정답 중심에서 R(px) 이내"로 매칭한다. `report.py --center-r 5 --tag _v2`(결과 `report_v2_*.json`). R=5px는 제안값이고 팀 합의 전이다.
  - R≥3px에서는 두 모델 모두 test 663개를 전부 찾고 오검출이 0이다. "찾았는가"는 포화이고, 구분은 R=1~2px(위치 정밀도)에서만 되며 이 구간에는 라벨 찍을 때의 오차가 섞여 있다.
  - IoU 0.5 기준 F1은 사실상 "중심을 2px 안에 맞췄는가"를 재고 있었다(R=2px 결과가 IoU 0.5 결과와 거의 같음).
  - 같은 test를 다른 규칙으로 재채점한 것이라 새 평가가 아니다. 오류가 0이라 부트스트랩 구간은 퇴화한다.
- **점 제거 실험**(`src/minyeop/dot_removal/remove_dot.py`, 결과 `runs/minyeop/03_dot_removal_v1/`): test 정답 663개의 어두운 점을 보간으로 지운 뒤 두 모델에 다시 넣어, 같은 자리(중심 5px 이내)에서 검출이 남는지 본다.

| 방식 | 제거 크기 | Faster R-CNN (≥0.95) | YOLOv3-tiny (≥0.06) |
|---|---|---|---|
| 원본 | | 663 | 663 |
| Navier-Stokes + 노이즈 | 9x9 / 13x13 / 17x17 | 38 / 2 / 1 | 249 / 198 / 92 |
| 평균 보간(노이즈 없음) | 9x9 / 13x13 / 17x17 | 39 / 2 / 0 | 38 / 1 / 0 |
| 평균 보간 + 노이즈 | 13x13 | 1 | 118 |

  - 점을 충분히 지우면 두 모델의 검출이 거의 0이 된다 → 결함의 점 자체가 핵심 단서이고, 위치·맥락만 보는 지름길 가설은 약해졌다.
  - YOLO에 남는 약한 반응(0.06~0.15)은 보간에 더한 노이즈 무늬 때문이었다(노이즈를 빼면 1개, 평균 보간에 노이즈를 더하면 118개).
  - 한계: 평평한 패치는 학습 때 본 적 없는 모양이고, 크기를 키우면 점 주변의 막대 일부도 지워진다. 시드 하나, test만 사용. 점이 없는 자리의 기준 대비, 점을 새로 그려 넣는 양성 대조군은 하지 않았다.

### 아직 없는 것
- **조건별 재현율** (호기·해상도·대비·배경·크기): `data/conditions.csv`로 계산해야 하지만 아직 계산 코드가 없음. 크기 구간은 공식 라벨 박스만 사용할 것 ([metrics.md](metrics.md) 참고)
- **중심 거리 기준 매칭**: IoU 0.5는 작은 박스에서 크기 차이에 민감해 점수가 포화된다. 중심 거리로 보는 평가를 병행할지는 팀 결정이 필요하다 (미정)
- **오류 사례 분석** (FN/FP 이미지 모음): Faster R-CNN은 `runs/minyeop/01_frcnn_r50fpn_min640_v1/figures/{val,test}/`에 그림이 있음
- 모델 간 차이의 **쌍체 부트스트랩** (같은 이미지에서 두 모델의 차이를 재표집): 모델이 둘 이상 생기면 추가

## 이전 문서 복구
지운 문서는 Git 이력에 남아 있습니다. 마지막으로 존재한 커밋은 `d151bfa`입니다.
```bash
git show d151bfa:docs/<파일명>.md          # 내용 보기
git checkout d151bfa -- docs/<파일명>.md   # 복구
```

| 이전 문서 | 내용 |
|---|---|
| `dataset.md` | 데이터 현황, 호기별 통계, **장비 표시(색 박스) 문제**, 라벨 500장, 표시 제거 방식 비교 |
| `original_folder_inventory.md` | 원본 폴더 전수 조사 |
| `troubleshooting.md` | 환경·코드 호환 문제와 해결 기록 |
| `experiment_results.md` | 실험 설정과 결과 (삭제 전 데이터 기준) |
| `evaluation.md` | 대회 평가 기준(배점) |
| `strategy.md` | 우승 전략 검토 |
| `structure.md` | 디렉터리 구조와 개편 내역 |

## 참고
- 다른 문서에는 위 문서로 가는 링크가 없고, 필요한 곳에는 `git show d151bfa:docs/<파일>` 명령으로 적어 두었습니다.
- 새로 겪은 문제와 해결은 새 문서에 기록하세요.
