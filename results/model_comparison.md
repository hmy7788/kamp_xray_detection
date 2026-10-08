| 모델 | mAP50 | mAP50-95 | Precision | Recall | F1 | 임계값(val) | FPS (CPU 4스레드) | TP/FP/FN |
|---|---|---|---|---|---|---|---|---|
| YOLOv3-tiny (베이스라인) | 0.9812 | 0.5579 | 0.9894 | 0.9894 | 0.9894 | 0.0753 | 14.0 | 656/7/7 |
| D-FINE-N (최종 모델) | 0.9886 | 0.6550 | 0.9880 | 0.9894 | 0.9887 | 0.7114 | 10.3 | 656/8/7 |
| Faster R-CNN (ResNet-50 FPN) | 0.9877 | 0.6114 | 0.9894 | 0.9894 | 0.9894 | 0.9500 | 1.9 | 656/7/7 |
| Faster R-CNN (MobileNetV3 FPN) | 0.9846 | 0.5587 | 0.9864 | 0.9864 | 0.9864 | 0.9300 | 10.0 | 654/9/9 |
| YOLO26n (입력 640) | 0.9872 | 0.6214 | 0.9910 | 0.9910 | 0.9910 | 0.7417 | 26.8 | 657/6/6 |
| RT-DETR-l | 0.9866 | 0.6387 | 0.9879 | 0.9879 | 0.9879 | 0.7400 | 2.8 | 655/8/8 |

- test 396장, 정답 상자 663개. 임계값은 모델별로 val F1 최대값으로 정해 test에 그대로 적용.
- 출처: results/comparison_models/six_models/summary.json; results/comparison_models/yolo26n_img640/common_eval_report_test.json; results/dfine_n/eval_report_test.json; results/yolov3_tiny/eval_report_test.json
