# runs/minyeop (허민엽)

허민엽의 실험 결과 폴더입니다. **본인만 수정합니다.**

## 규칙
- 실험 하나 = 폴더 하나: `<번호>_<모델>_<설정>/` (예: `01_yolov3tiny_nomark`)
- 번호는 이 폴더 안에서 이어 붙입니다. 이미 있는 이름은 덮어쓰지 않습니다.
- 끝까지 돌지 못한 실험은 최종 결과로 쓰지 않고 이름을 `_incomplete_...`로 바꿉니다.
- 폴더 구성과 Git에 올리는 것/제외하는 것은 [../README.md](../README.md)를 보세요.

## 실험 목록
| 실험 | 모델 / 설정 | 결과 (val에서 임계값 고정) | 비고 |
|---|---|---|---|
| `01_frcnn_r50fpn_min640_v1` | Faster R-CNN R50-FPN, COCO 사전학습, 입력 640, 기본 앵커, 20 epoch(선택 epoch 7) | test F1 0.989, 공식 라벨만 0.964 (val F1 0.992), 임계값 0.95 | 표: [docs/experiments.md](../../docs/experiments.md), 그림: `01_frcnn_r50fpn_min640_v1/figures/` |
| `02_yolov3tiny_img640_v1` | YOLOv3-tiny(ultralytics 2020), COCO 사전학습, 입력 640, 100 epoch(선택 59번째) | test F1 0.989, 공식 라벨만 0.964, 임계값 0.06 | 코드 `src/minyeop/yolov3_tiny/`. `train.log`와 학습 배치 샘플 이미지는 Git 제외 |
| `03_dot_removal_v1` | 점 제거 분석(학습 아님): 보간 방식·크기별 `ns_half*`, `mean_half*`, `meannoise_half6` | 점을 충분히 지우면 두 모델 검출이 거의 0 | 코드 `src/minyeop/dot_removal/`, 해석은 `docs/README.md` |
| `04_frcnn_mobv3_min640_anc16_v1` | Faster R-CNN MobileNetV3-Large FPN, COCO 사전학습, 입력 640, 앵커 16~256, 20 epoch(선택 18) | test F1 0.986, 공식 라벨만 0.959, 임계값 0.93 | 코드 `src/minyeop/faster_rcnn_mobilenet/`. ResNet-50 대비 학습 20분, 추론 약 4배 빠름 |
