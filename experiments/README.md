# experiments/ — 비교 모델과 분석 실험의 원 코드 (참고용)

보고서 2장(비교 모델)과 3장·5장(오류분석, 합성 이물질 실험)의 수치를 만든 **원 실험 코드**를 그대로 보존한 폴더다.
최종 모델·베이스라인의 재현 파이프라인(`scripts/`, `src/kamp_xray/`)과 달리 **경로가 원 실험 저장소 구조 기준**이라
이 폴더에서 바로 실행되지 않을 수 있다. 결과는 모두 `results/`에 저장되어 있으므로, 수치 확인은 결과 파일로 한다.

모든 모델은 같은 데이터(`data/`, manifest sha256 `1942bf34...`)와 같은 분할로 학습했고,
6개 모델의 test 지표는 같은 채점 규칙(IoU 0.5 탐욕 매칭, VOC AP, val F1 최대 임계값)으로 계산했다.

## 비교 모델 (`comparison_models/`)

| 모델 | 코드 | 결과 | 작성 |
|---|---|---|---|
| Faster R-CNN (ResNet-50 FPN) | `faster_rcnn/` (torchvision) | `results/comparison_models/faster_rcnn_r50/` | 허민엽 |
| Faster R-CNN (MobileNetV3 FPN) | `faster_rcnn_mobilenet/` | `results/comparison_models/faster_rcnn_mobilenetv3/` | 허민엽 |
| RT-DETR-l | `rtdetr/` (ultralytics) | `results/comparison_models/rtdetr_l/` | 정연창 |
| YOLO26n (입력 1024) | `yolo26/` (ultralytics, 실험 하네스) | `results/comparison_models/yolo26n/` | 이총 |

6개 모델 통합 채점: `analysis/extra_models/eval_extra.py` → `results/comparison_models/six_models/summary.json`.
비교 모델 실행에는 `requirements.txt` 외에 `ultralytics==8.4.174`가 필요하다.

## 분석 실험 (`analysis/`)

| 실험 | 보고서 | 코드 | 결과 |
|---|---|---|---|
| 합성 이물질 데이터 생성 (막대 안, 대비 4단계 × 크기 3단계, seed 42) | 3.3절, 5.1절 | `scripts/synth_eval.py` (바로 실행 가능) | `results/analysis/08_synth_eval_v1/` |
| 합성 이물질 6개 모델 평가 (원래 위치 / 무작위 위치) | 3.3절 | `synth_eval/`, `synth_insert/`, `extra_models/` | `results/analysis/11_synth_insert_6models_v1/` |
| 이물질 제거 영상(가짜 정상)의 오검출 | 3.2절, 5.1절 | `dot_removal/`, `fake_normal/` | `results/analysis/03_dot_removal_v1/`, `05_fake_normal_v1/` |
| 호기·해상도별 성능 | 3.1절 | `group_stats/`, `extra_models/group_stats_all.py` | `results/analysis/12_group_stats_6models_v1/` |
| 전처리 재현 검증 | 1.3절 | `scripts/verify_preprocess.py` (바로 실행 가능) | `results/analysis/10_preprocess_check_v1/` |
| 속도 재측정 (6개 모델) | 2.3절, 4.2절, 5.2절 | `extra_models/fps_all.py` | `results/comparison_models/six_models/fps_all.json` |

합성 이물질 데이터는 `python scripts/synth_eval.py --split test --out data_synth/test --verify`로 다시 만들고
`scripts/synth_eval_reference/`의 기준 해시와 같은지 확인할 수 있다.
