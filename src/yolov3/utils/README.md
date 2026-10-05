# src/yolov3/utils

YOLOv3 학습·추론이 쓰는 보조 코드입니다. 이 폴더도 ultralytics/yolov3의 수정본이라 **공유 영역**입니다. 출처와 수정 내역은 [../README.md](../README.md)를 보세요.

| 파일 | 역할 |
|---|---|
| `datasets.py` | 이미지·라벨 불러오기, 증강. **한글 경로용 `imread`/`imwrite` 헬퍼**를 추가해 둠 |
| `utils.py` | 손실 계산, NMS, 좌표 변환, 지표 계산 |
| `layers.py`, `torch_utils.py` | 네트워크 층과 PyTorch 보조 함수 |
| `parse_config.py` | `.cfg`, `.data` 파일 읽기 |
| `google_utils.py` | 가중치 다운로드 보조 (원본 링크가 낡아 쓰지 않음) |
| `adabound.py`, `evolve.sh`, `gcp.sh` | 원본에 있던 옵티마이저·실험 보조 파일 (사용하지 않음) |
