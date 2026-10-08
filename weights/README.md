# weights 폴더

학습을 마친 두 모델의 가중치가 들어 있습니다. 한 번에 실행하기 명령은 이 가중치로 예측과 채점을 합니다.

| 위치 | 내용 | 크기 |
|---|---|---|
| dfine_n/best | 최종 모델 D-FINE-N의 가중치와 설정 파일 세 개 | 약 15MB |
| yolov3_tiny/best.pt | 베이스라인 YOLOv3-tiny의 가중치 | 약 69MB |
| pretrained | 학습을 처음부터 할 때 쓰는 YOLOv3-tiny 사전학습 가중치가 저장되는 곳 | 약 35MB |

- 두 가중치는 이 저장소의 학습 코드와 configs 폴더의 설정으로 학습한 것입니다. 학습 기록은 results 폴더의 original_run에 있습니다.
- D-FINE-N은 최대 50 에폭 중 43 에폭에서 학습이 멈췄고, 검증용 영상 점수가 가장 좋았던 33 에폭의 가중치를 골랐습니다.
- YOLOv3-tiny는 100 에폭을 학습했고, 검증용 영상 점수가 가장 좋았던 시점의 가중치를 골랐습니다.
- D-FINE-N의 사전학습 가중치는 학습을 시작할 때 Hugging Face에서 자동으로 내려받습니다.
- YOLOv3-tiny의 사전학습 가중치는 scripts 폴더의 download_pretrained.py가 Darknet 공식 COCO 가중치를 내려받아 이 저장소에서 쓰는 형식으로 바꿔 저장합니다.
