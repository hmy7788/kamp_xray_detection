# weights

| 위치 | 내용 | 크기 |
|---|---|---|
| dfine_n/best | 최종 모델 D-FINE-N 가중치와 설정 파일 | 약 15MB |
| yolov3_tiny/best.pt | 베이스라인 YOLOv3-tiny 가중치 | 약 69MB |
| pretrained | 학습 시 사용하는 YOLOv3-tiny COCO 사전학습 가중치 저장 위치 | 약 35MB |

- 두 가중치는 이 저장소의 학습 코드와 configs 폴더의 설정으로 학습한 것이며, 학습 기록은 results 폴더의 original_run에 있음
- D-FINE-N은 최대 50 에폭 중 43 에폭에서 조기 종료되었고, 검증 데이터 성능이 가장 높은 33 에폭의 가중치를 사용함
- YOLOv3-tiny는 100 에폭 학습 후 검증 데이터 성능이 가장 높은 시점의 가중치를 사용함
- D-FINE-N 사전학습 가중치는 학습 시작 시 Hugging Face에서 자동으로 내려받음
- YOLOv3-tiny 사전학습 가중치는 scripts/download_pretrained.py가 Darknet 공식 COCO 가중치를 내려받아 변환함
