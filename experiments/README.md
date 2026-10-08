# experiments 폴더

보고서 2장의 비교 모델과 3장, 5장의 분석 실험을 만든 코드를 원래 모습 그대로 보관한 폴더입니다.
팀원 각자의 실험 폴더 구조에 맞춰 작성된 코드라서 이 저장소에서 그대로 실행되지 않을 수 있습니다.
결과는 모두 results 폴더에 저장되어 있습니다.

모든 모델은 data 폴더의 같은 데이터와 같은 분할로 학습했고, 같은 채점 방식으로 평가했습니다.

## 비교 모델

| 모델 | 코드 위치 | 결과 위치 | 작성 |
|---|---|---|---|
| Faster R-CNN, ResNet-50 백본 | comparison_models/faster_rcnn | results/comparison_models/faster_rcnn_r50 | 허민엽 |
| Faster R-CNN, MobileNetV3 백본 | comparison_models/faster_rcnn_mobilenet | results/comparison_models/faster_rcnn_mobilenetv3 | 허민엽 |
| RT-DETR-l | comparison_models/rtdetr | results/comparison_models/rtdetr_l | 정연창 |
| YOLO26n | comparison_models/yolo26 | results/comparison_models/yolo26n_img640, yolo26n_img1024 | 이총 |

- YOLO26n은 입력 크기 640과 1024로 각각 학습했습니다. 비교표에는 보고서에서 기본으로 정한 입력 640 모델의 점수를 씁니다.
- comparison_models/yolo26 폴더의 rescore_common.py는 YOLO26n을 다른 모델과 같은 채점 코드로 다시 채점하는 파일입니다.
- 비교 모델을 실행하려면 requirements.txt의 패키지에 더해 ultralytics 8.4.174가 필요합니다.

## 분석 실험

| 실험 | 보고서 | 코드 위치 | 결과 위치 |
|---|---|---|---|
| 합성 이물질 데이터 만들기 | 3.3절, 5.1절 | scripts/synth_eval.py | results/analysis/08_synth_eval_v1 |
| 합성 이물질로 여섯 모델 평가 | 3.3절 | analysis/synth_eval, analysis/synth_insert, analysis/extra_models | results/analysis/11_synth_insert_6models_v1 |
| 이물질을 지운 영상의 오검출 | 3.2절, 5.1절 | analysis/dot_removal, analysis/fake_normal | results/analysis/03_dot_removal_v1, 05_fake_normal_v1 |
| 장비별, 해상도별 성능 | 3.1절 | analysis/group_stats, analysis/extra_models | results/analysis/12_group_stats_6models_v1 |
| 전처리 재현 검증 | 1.3절 | scripts/verify_preprocess.py | results/analysis/10_preprocess_check_v1 |
| 여섯 모델 속도 측정 | 2.3절, 5.2절 | analysis/extra_models/fps_all.py | results/comparison_models/six_models/fps_all.json |

합성 이물질 데이터는 아래 명령으로 다시 만들 수 있고, 만든 데이터가 보고서에 쓴 것과 똑같은지 함께 검사합니다.

```powershell
python scripts/synth_eval.py --split test --out outputs/data_synth/test --verify
```
