# weights/ — 가중치

| 경로 | 내용 | 크기 | 쓰는 곳 |
|---|---|---|---|
| `dfine_n/best/` | **최종 모델 D-FINE-N** 제출 가중치 (Hugging Face 형식: `config.json`, `model.safetensors`, `preprocessor_config.json`) | 약 15MB | 추론·채점 (`run_all` 기본) |
| `yolov3_tiny/best.pt` | **베이스라인 YOLOv3-tiny** 제출 가중치 | 약 69MB | 추론·채점 (`run_all` 기본) |
| `pretrained/yolov3-tiny.pt` | YOLOv3-tiny COCO 사전학습 가중치 | 약 35MB | 학습할 때만 (`scripts/download_pretrained.py`가 만든다) |

- 제출 가중치는 `scripts/train.py`와 같은 코드·설정(`configs/`)으로 학습한 것이다. 원 학습 기록은 `results/<모델>/original_run/`.
  - D-FINE-N: 43 epoch에서 조기 종료, val 기준 최적 33 epoch의 EMA 가중치
  - YOLOv3-tiny: 100 epoch, val 0.99×mAP50 + 0.01×R 최고 epoch
- D-FINE-N의 COCO 사전학습 가중치(`ustc-community/dfine-nano-coco`)는 학습을 시작할 때 Hugging Face에서 자동으로 받는다.
- YOLOv3-tiny 사전학습 가중치: 원본 YOLOv3 코드가 쓰던 Google Drive 링크가 만료되어, Darknet 공식 COCO 가중치
  `https://pjreddie.com/media/files/yolov3-tiny.weights`를 받아 이 저장소의 YOLOv3 형식(.pt)으로 변환한다.
  인터넷이 막힌 환경에서는 그 파일을 `pretrained/yolov3-tiny.weights`로 넣고 `python scripts/download_pretrained.py`를 실행하면 변환만 한다.
