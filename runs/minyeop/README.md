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
| `01_frcnn_r50fpn_min640` | Faster R-CNN R50-FPN, COCO 사전학습, 입력 640, 기본 앵커, 20 epoch(선택 epoch 7) | test F1 0.989, 공식 라벨만 0.964 (val F1 0.992), 임계값 0.95 | 표: [docs/experiments.md](../../docs/experiments.md), 그림: `01_frcnn_r50fpn_min640/figures/` |
