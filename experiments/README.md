# experiments

보고서 2장의 비교 모델과 3장, 5장의 분석 실험에 사용한 원본 코드임. 결과는 results 폴더에 저장되어 있음

- 팀원별 실험 폴더 구조 기준으로 작성된 코드로, 경로 수정 없이 보존함
- 모든 모델은 data 폴더의 동일한 데이터와 분할로 학습하고 동일한 채점 방식으로 평가함
- 비교 모델 실행에는 requirements.txt 외에 ultralytics 8.4.174가 필요함

## 비교 모델

| 모델 | 코드 | 결과 | 담당 |
|---|---|---|---|
| Faster R-CNN, ResNet-50 FPN | comparison_models/faster_rcnn | results/comparison_models/faster_rcnn_r50 | 허민엽 |
| Faster R-CNN, MobileNetV3 FPN | comparison_models/faster_rcnn_mobilenet | results/comparison_models/faster_rcnn_mobilenetv3 | 허민엽 |
| RT-DETR-l | comparison_models/rtdetr | results/comparison_models/rtdetr_l | 정연창 |
| YOLO26n | comparison_models/yolo26 | results/comparison_models/yolo26n_img640, yolo26n_img1024 | 이총 |

- YOLO26n은 입력 640과 1024로 각각 학습했으며, 성능표에는 기본 입력 크기인 640 모델을 사용함
- comparison_models/yolo26/rescore_common.py로 YOLO26n을 공통 채점 코드로 다시 채점함

## 분석 실험

| 실험 | 보고서 | 코드 | 결과 |
|---|---|---|---|
| 합성 이물질 데이터 생성 | 3.3절, 5.1절 | scripts/synth_eval.py | results/analysis/08_synth_eval_v1 |
| 합성 이물질 6개 모델 평가 | 3.3절 | analysis/synth_eval, synth_insert, extra_models | results/analysis/11_synth_insert_6models_v1 |
| 이물질 제거 영상 오검출 | 3.2절, 5.1절 | analysis/dot_removal, fake_normal | results/analysis/03_dot_removal_v1, 05_fake_normal_v1 |
| 호기별, 해상도별 성능 | 3.1절 | analysis/group_stats, extra_models | results/analysis/12_group_stats_6models_v1 |
| 전처리 재현 검증 | 1.3절 | scripts/verify_preprocess.py | results/analysis/10_preprocess_check_v1 |
| 6개 모델 속도 측정 | 2.3절, 5.2절 | analysis/extra_models/fps_all.py | results/comparison_models/six_models/fps_all.json |

합성 이물질 데이터는 아래 명령으로 다시 생성하며, 생성 결과가 보고서에 사용한 데이터와 같은지 함께 검사함

```bash
python scripts/synth_eval.py --split test --out outputs/data_synth/test --verify
```
